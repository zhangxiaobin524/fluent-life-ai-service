"""
LLM-as-a-Judge 评测核心实现
"""

import os
import json
import re
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from openai import OpenAI

from .test_cases import (
    TEST_CASES, 
    EVALUATION_DIMENSIONS, 
    EVALUATION_CRITERIA
)


@dataclass
class EvaluationResult:
    """单次评测结果"""
    module: str
    test_case_id: str
    test_case_name: str
    dimension: str
    score: float  # 0-10
    weight: int   # 权重百分比
    reason: str
    suggestion: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "module": self.module,
            "test_case_id": self.test_case_id,
            "test_case_name": self.test_case_name,
            "dimension": self.dimension,
            "score": self.score,
            "weight": self.weight,
            "weighted_score": round(self.score * self.weight / 100, 2),
            "reason": self.reason,
            "suggestion": self.suggestion,
            "timestamp": self.timestamp
        }


class LLMJudge:
    """LLM 裁判 - 使用 DeepSeek API 进行评测"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("需要提供 DeepSeek API Key 或设置 DEEPSEEK_API_KEY 环境变量")
        
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.model = "deepseek-chat"
    
    def _build_judge_prompt(
        self,
        module: str,
        dimension: str,
        test_case: Dict,
        ai_output: Any
    ) -> str:
        """构建裁判 Prompt"""
        
        criteria = EVALUATION_CRITERIA.get(module, {}).get(dimension, "无详细标准")
        
        prompt = f"""你是专业的 AI 系统输出质量评估专家。请对以下 AI 系统的输出进行客观、严格的评分。

【评测任务】
- 被测模块: {module}
- 评测维度: {dimension}
- 测试用例: {test_case.get('name', '未命名')}

【评分标准】
{criteria}

【输入信息】
用户输入: {test_case.get('user_message', json.dumps(test_case.get('user_profile', {}), ensure_ascii=False))}

【AI 系统输出】
```json
{json.dumps(ai_output, ensure_ascii=False, indent=2)[:2000]}
```

【预期行为参考】
{json.dumps(test_case.get('expected', {}), ensure_ascii=False, indent=2)}

【评分要求】
1. 严格按评分标准打分，范围 0.0-10.0，保留一位小数
2. 评分理由需具体指出输出中的优点和问题（80-150字）
3. 如有改进空间，给出具体建议（如有）

【输出格式】
必须严格按以下格式输出，不要添加其他内容：

分数: <数字>
理由: <理由>
建议: <建议或无>
"""
        return prompt
    
    def evaluate(
        self,
        module: str,
        dimension: str,
        test_case: Dict,
        ai_output: Any
    ) -> EvaluationResult:
        """执行单次评测"""
        
        prompt = self._build_judge_prompt(module, dimension, test_case, ai_output)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # 低温度保证评分一致性
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            
            # 解析结果
            score = self._extract_score(result_text)
            reason = self._extract_field(result_text, "理由")
            suggestion = self._extract_field(result_text, "建议")
            
            weight = EVALUATION_DIMENSIONS.get(module, {}).get(dimension, 25)
            
            return EvaluationResult(
                module=module,
                test_case_id=test_case.get("id", "unknown"),
                test_case_name=test_case.get("name", "未命名"),
                dimension=dimension,
                score=score,
                weight=weight,
                reason=reason,
                suggestion=suggestion
            )
            
        except Exception as e:
            # 评测失败返回 0 分并记录错误
            return EvaluationResult(
                module=module,
                test_case_id=test_case.get("id", "unknown"),
                test_case_name=test_case.get("name", "未命名"),
                dimension=dimension,
                score=0.0,
                weight=EVALUATION_DIMENSIONS.get(module, {}).get(dimension, 25),
                reason=f"评测过程出错: {str(e)}",
                suggestion="请检查 AI 服务是否正常运行"
            )
    
    def _extract_score(self, text: str) -> float:
        """从裁判输出中提取分数"""
        # 匹配 "分数: 8.5" 或 "分数：8.5"
        patterns = [
            r'分数[：:]\s*(\d+\.?\d*)',
            r'score[：:]\s*(\d+\.?\d*)',
            r'(\d+\.?\d+)\s*分'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 10.0)  # 限制在 0-10 范围内
        
        # 如果没找到，尝试找任何数字
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            return min(max(float(numbers[0]), 0.0), 10.0)
        
        return 0.0
    
    def _extract_field(self, text: str, field_name: str) -> str:
        """提取字段内容"""
        pattern = rf'{field_name}[：:]\s*(.+?)(?=\n|$|(?:[^\n]{{2,}}[：:]))'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "未提取到内容"


class AIEvaluator:
    """AI 评测器主类"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.judge = LLMJudge(api_key)
        self.results: List[EvaluationResult] = []
    
    async def evaluate_module_async(
        self,
        module: str,
        test_cases: List[Dict],
        ai_callable: Callable[[Dict], Any]
    ) -> Dict[str, Any]:
        """异步评测一个模块的所有测试用例"""
        
        print(f"\n{'='*60}")
        print(f"开始评测模块: {module}")
        print(f"{'='*60}")
        
        dimensions = EVALUATION_DIMENSIONS.get(module, {})
        module_results: List[EvaluationResult] = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[测试用例 {i}/{len(test_cases)}] {test_case.get('name', '未命名')}")
            
            # 1. 调用被测 AI
            try:
                if asyncio.iscoroutinefunction(ai_callable):
                    ai_output = await ai_callable(test_case)
                else:
                    ai_output = ai_callable(test_case)
            except Exception as e:
                print(f"  ❌ AI 调用失败: {e}")
                ai_output = {"error": str(e)}
            
            # 2. 对每个维度进行评测
            for dimension in dimensions.keys():
                print(f"  评测维度: {dimension}...", end=" ")
                
                result = self.judge.evaluate(module, dimension, test_case, ai_output)
                module_results.append(result)
                self.results.append(result)
                
                print(f"得分: {result.score}")
        
        # 计算模块综合得分
        return self._calculate_module_summary(module, module_results)
    
    def evaluate_module(
        self,
        module: str,
        test_cases: List[Dict],
        ai_callable: Callable[[Dict], Any]
    ) -> Dict[str, Any]:
        """同步评测接口"""
        return asyncio.run(self.evaluate_module_async(module, test_cases, ai_callable))
    
    def _calculate_module_summary(
        self,
        module: str,
        results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """计算模块评测摘要"""
        
        dimensions = EVALUATION_DIMENSIONS.get(module, {})
        
        # 按维度统计
        dimension_stats = {}
        for dim_name in dimensions.keys():
            dim_results = [r for r in results if r.dimension == dim_name]
            if dim_results:
                scores = [r.score for r in dim_results]
                avg_score = sum(scores) / len(scores)
                weight = dimensions[dim_name]
                
                dimension_stats[dim_name] = {
                    "avg_score": round(avg_score, 2),
                    "weight": weight,
                    "weighted_score": round(avg_score * weight / 100, 2),
                    "min_score": round(min(scores), 2),
                    "max_score": round(max(scores), 2),
                    "test_count": len(scores)
                }
        
        # 计算总分
        total_score = sum(s["weighted_score"] for s in dimension_stats.values())
        
        # 找出低分项目（< 6分）
        low_scores = [
            r.to_dict() for r in results 
            if r.score < 6.0
        ]
        
        summary = {
            "module": module,
            "total_score": round(total_score, 2),
            "grade": self._score_to_grade(total_score),
            "dimensions": dimension_stats,
            "total_test_cases": len(set(r.test_case_id for r in results)),
            "total_evaluations": len(results),
            "low_score_items": low_scores[:5],  # 最多显示5个
            "all_results": [r.to_dict() for r in results]
        }
        
        print(f"\n{'='*60}")
        print(f"模块 {module} 评测完成")
        print(f"综合得分: {total_score:.2f}/10.0 ({summary['grade']})")
        print(f"{'='*60}")
        
        return summary
    
    def _score_to_grade(self, score: float) -> str:
        """分数转等级"""
        if score >= 9.0:
            return "优秀 (A+)"
        elif score >= 8.0:
            return "良好 (A)"
        elif score >= 7.0:
            return "中等 (B)"
        elif score >= 6.0:
            return "及格 (C)"
        elif score >= 5.0:
            return "较差 (D)"
        else:
            return "不及格 (F)"
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """生成完整评测报告"""
        
        report_lines = []
        report_lines.append("# AI 模块质量评测报告")
        report_lines.append("")
        report_lines.append(f"评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"评测模块数: {len(set(r.module for r in self.results))}")
        report_lines.append(f"总评测次数: {len(self.results)}")
        report_lines.append("")
        
        # 按模块分组
        modules = {}
        for r in self.results:
            modules.setdefault(r.module, []).append(r)
        
        for module, results in modules.items():
            summary = self._calculate_module_summary(module, results)
            
            report_lines.append(f"## {module} 模块")
            report_lines.append("")
            report_lines.append(f"**综合得分: {summary['total_score']}/10.0 ({summary['grade']})**")
            report_lines.append("")
            
            # 维度详情表格
            report_lines.append("### 各维度评分")
            report_lines.append("")
            report_lines.append("| 评测维度 | 平均得分 | 权重 | 加权得分 | 最高 | 最低 |")
            report_lines.append("|---------|---------|------|---------|------|------|")
            
            for dim_name, stats in summary['dimensions'].items():
                report_lines.append(
                    f"| {dim_name} | {stats['avg_score']} | {stats['weight']}% | "
                    f"{stats['weighted_score']} | {stats['max_score']} | {stats['min_score']} |"
                )
            
            report_lines.append("")
            
            # 低分项目
            if summary['low_score_items']:
                report_lines.append("### 需要改进的项目")
                report_lines.append("")
                for item in summary['low_score_items']:
                    report_lines.append(f"- **{item['test_case_name']}** - {item['dimension']}: {item['score']}分")
                    report_lines.append(f"  - 原因: {item['reason']}")
                    if item['suggestion'] and item['suggestion'] != "无":
                        report_lines.append(f"  - 建议: {item['suggestion']}")
                report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n报告已保存: {output_file}")
        
        return report


# ========================================
# 辅助函数：快速评测
# ========================================

def quick_evaluate_expert_team(
    api_key: Optional[str] = None,
    workflow_func: Optional[Callable] = None
) -> Dict[str, Any]:
    """快速评测 ExpertTeam 模块"""
    
    evaluator = AIEvaluator(api_key)
    test_cases = TEST_CASES["expert_team"]
    
    # 默认使用 mock 调用
    if workflow_func is None:
        def mock_call(case):
            # 这里应该调用真实的 ExpertTeam workflow
            # 返回模拟数据用于测试
            return {
                "question_type": case.get("expected", {}).get("question_type", "general"),
                "experts_involved": case.get("expected", {}).get("should_involve", []),
                "final_response": "这是模拟的专家团队回复",
                "execution_path": ["router", "expert"]
            }
        workflow_func = mock_call
    
    return evaluator.evaluate_module("expert_team", test_cases, workflow_func)


def quick_evaluate_training_plan(
    api_key: Optional[str] = None,
    workflow_func: Optional[Callable] = None
) -> Dict[str, Any]:
    """快速评测 TrainingPlan 模块"""
    
    evaluator = AIEvaluator(api_key)
    test_cases = TEST_CASES["training_plan"]
    
    if workflow_func is None:
        def mock_call(case):
            profile = case.get("user_profile", {})
            return {
                "title": "7天口吃矫正训练计划",
                "duration_days": 7,
                "daily_time": profile.get("available_time", "30分钟"),
                "exercises": ["呼吸训练", "气流法", "慢速朗读"],
                "difficulty": profile.get("level", "初级")
            }
        workflow_func = mock_call
    
    return evaluator.evaluate_module("training_plan", test_cases, workflow_func)
