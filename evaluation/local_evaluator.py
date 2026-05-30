"""
本地评测器 - 不依赖外部 API，使用规则-based 评分
用于 API 余额不足时的本地测试
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime

from .test_cases import TEST_CASES, EVALUATION_DIMENSIONS


@dataclass
class LocalEvaluationResult:
    """本地评测结果"""
    module: str
    test_case_id: str
    test_case_name: str
    dimension: str
    score: float
    weight: int
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


class LocalJudge:
    """本地裁判 - 基于规则评分，不调用 API"""
    
    def evaluate(
        self,
        module: str,
        dimension: str,
        test_case: Dict,
        ai_output: Any
    ) -> LocalEvaluationResult:
        """基于规则的本地评分"""
        
        expected = test_case.get("expected", {})
        score = 5.0  # 默认中等分数
        reason = ""
        suggestion = ""
        
        if module == "expert_team":
            score, reason, suggestion = self._evaluate_expert_team(
                dimension, expected, ai_output
            )
        elif module == "training_plan":
            score, reason, suggestion = self._evaluate_training_plan(
                dimension, expected, ai_output
            )
        
        weight = EVALUATION_DIMENSIONS.get(module, {}).get(dimension, 25)
        
        return LocalEvaluationResult(
            module=module,
            test_case_id=test_case.get("id", "unknown"),
            test_case_name=test_case.get("name", "未命名"),
            dimension=dimension,
            score=round(score, 1),
            weight=weight,
            reason=reason,
            suggestion=suggestion
        )
    
    def _evaluate_expert_team(self, dimension: str, expected: Dict, output: Dict) -> tuple:
        """评测 ExpertTeam"""
        
        if dimension == "路由准确性":
            expected_type = expected.get("question_type", "")
            actual_type = output.get("question_type", "")
            
            if expected_type == actual_type:
                return 9.0, f"正确识别问题类型为 {actual_type}", "保持当前路由逻辑"
            elif actual_type == "general" and expected_type != "general":
                return 5.0, f"期望识别为 {expected_type}，但实际路由为通用类型", "优化路由模型，提升识别精度"
            else:
                return 3.0, f"路由错误：期望 {expected_type}，实际 {actual_type}", "检查路由逻辑，增加训练数据"
        
        elif dimension == "专家协作度":
            expected_experts = set(expected.get("should_involve", []))
            actual_experts = set(output.get("experts_involved", []))
            
            if expected_experts == actual_experts:
                return 9.0, "专家调用完全匹配预期", "无"
            elif expected_experts & actual_experts:  # 有交集
                return 6.0, f"部分专家匹配：期望 {expected_experts}，实际 {actual_experts}", "完善专家协作逻辑"
            else:
                return 3.0, f"专家调用偏差较大", "重新设计专家调用策略"
        
        elif dimension == "回答完整性":
            response = output.get("final_response", "")
            key_elements = expected.get("key_elements", [])
            
            if len(response) > 100 and key_elements:
                return 8.0, "回答较为完整，包含关键信息", "可适当增加细节"
            elif len(response) > 50:
                return 6.0, "回答基本完整但略显简略", "增加内容深度"
            else:
                return 4.0, "回答过于简短，信息不足", "大幅扩展回答内容"
        
        elif dimension == "专业度":
            response = output.get("final_response", "")
            if "口吃" in response or "矫正" in response or "训练" in response:
                return 8.5, "回答体现口吃矫正专业领域知识", "保持专业性"
            else:
                return 6.0, "回答专业术语使用较少", "增加口吃矫正专业术语"
        
        return 5.0, "默认评分", "无"
    
    def _evaluate_training_plan(self, dimension: str, expected: Dict, output: Dict) -> tuple:
        """评测 TrainingPlan"""
        
        if dimension == "计划可行性":
            daily_time = output.get("daily_time", "")
            expected_time = expected.get("daily_time", "")
            
            if daily_time == expected_time:
                return 9.0, "每日训练时长符合用户可用时间", "无"
            elif daily_time:
                return 6.0, f"训练时长 {daily_time} 与用户期望 {expected_time} 有偏差", "调整时长匹配用户需求"
            else:
                return 4.0, "未明确训练时长", "补充时长信息"
        
        elif dimension == "个性化程度":
            exercises = output.get("exercises", [])
            expected_exercises = expected.get("should_include", [])
            personalization = output.get("personalization_note", "")
            
            if expected_exercises and any(e in str(exercises) for e in expected_exercises):
                score = 8.5 if personalization else 7.0
                reason = "训练项目符合用户类型" + ("，有个人化说明" if personalization else "，但缺少个人化说明")
                return score, reason, "增加更多个性化描述" if not personalization else "无"
            else:
                return 5.0, "训练项目与用户类型匹配度不高", "根据用户类型定制训练项目"
        
        elif dimension == "目标明确性":
            goals = output.get("goals", [])
            if len(goals) >= 3:
                return 8.5, f"设定了 {len(goals)} 个明确目标，层次清晰", "无"
            elif len(goals) >= 1:
                return 6.5, f"有 {len(goals)} 个目标，但可更详细", "增加阶段性目标"
            else:
                return 4.0, "缺少明确目标", "补充具体可量化目标"
        
        elif dimension == "进阶合理性":
            difficulty = output.get("difficulty", "")
            expected_difficulty = expected.get("difficulty", "")
            
            if difficulty == expected_difficulty:
                return 8.5, f"难度设置 {difficulty} 符合用户水平", "无"
            elif difficulty:
                return 6.0, f"难度 {difficulty} 与用户期望 {expected_difficulty} 略有差异", "调整难度匹配"
            else:
                return 5.0, "难度未明确", "明确难度等级"
        
        return 5.0, "默认评分", "无"


class LocalAIEvaluator:
    """本地 AI 评测器 - 不依赖外部 API"""
    
    def __init__(self):
        self.judge = LocalJudge()
        self.results: List[LocalEvaluationResult] = []
    
    async def evaluate_module_async(
        self,
        module: str,
        test_cases: List[Dict],
        ai_callable: Any
    ) -> Dict[str, Any]:
        """异步评测"""
        import asyncio
        
        print(f"\n{'='*60}")
        print(f"开始评测模块: {module} (本地模式)")
        print(f"{'='*60}")
        
        dimensions = EVALUATION_DIMENSIONS.get(module, {})
        module_results: List[LocalEvaluationResult] = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[测试用例 {i}/{len(test_cases)}] {test_case.get('name', '未命名')}")
            
            # 调用 AI
            try:
                if asyncio.iscoroutinefunction(ai_callable):
                    ai_output = await ai_callable(test_case)
                else:
                    ai_output = ai_callable(test_case)
            except Exception as e:
                print(f"  ❌ AI 调用失败: {e}")
                ai_output = {"error": str(e)}
            
            # 评测每个维度
            for dimension in dimensions.keys():
                print(f"  评测维度: {dimension}...", end=" ")
                
                result = self.judge.evaluate(module, dimension, test_case, ai_output)
                module_results.append(result)
                self.results.append(result)
                
                print(f"得分: {result.score}")
        
        return self._calculate_summary(module, module_results)
    
    def evaluate_module(self, module, test_cases, ai_callable):
        """同步接口"""
        import asyncio
        return asyncio.run(self.evaluate_module_async(module, test_cases, ai_callable))
    
    def _calculate_summary(self, module: str, results: List[LocalEvaluationResult]) -> Dict:
        """计算摘要"""
        dimensions = EVALUATION_DIMENSIONS.get(module, {})
        
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
        
        total_score = sum(s["weighted_score"] for s in dimension_stats.values())
        
        return {
            "module": module,
            "total_score": round(total_score, 2),
            "grade": self._score_to_grade(total_score),
            "dimensions": dimension_stats,
            "total_test_cases": len(set(r.test_case_id for r in results)),
            "total_evaluations": len(results),
            "all_results": [r.to_dict() for r in results]
        }
    
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
    
    def generate_report(self, output_file: str = None) -> str:
        """生成报告"""
        report_lines = ["# AI 模块质量评测报告 (本地评测)"]
        report_lines.append(f"\n评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("评测方式: 本地规则-based 评分 (无需 API)")
        
        modules = {}
        for r in self.results:
            modules.setdefault(r.module, []).append(r)
        
        for module, results in modules.items():
            summary = self._calculate_summary(module, results)
            report_lines.append(f"\n## {module} 模块")
            report_lines.append(f"**综合得分: {summary['total_score']}/10.0 ({summary['grade']})**")
            report_lines.append("\n| 评测维度 | 平均得分 | 权重 | 加权得分 |")
            report_lines.append("|---------|---------|------|---------|")
            
            for dim_name, stats in summary['dimensions'].items():
                report_lines.append(
                    f"| {dim_name} | {stats['avg_score']} | {stats['weight']}% | {stats['weighted_score']} |"
                )
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n报告已保存: {output_file}")
        
        return report


# 快速运行本地评测
def run_local_evaluation():
    """运行本地评测（无需 API）"""
    from evaluation.quick_test import mock_expert_team, mock_training_plan
    
    print("="*70)
    print(" 本地评测模式 (无需 DeepSeek API)")
    print("="*70)
    
    evaluator = LocalAIEvaluator()
    
    # 评测 ExpertTeam
    print("\n📋 评测 ExpertTeam...")
    expert_result = evaluator.evaluate_module(
        "expert_team",
        TEST_CASES["expert_team"][:5],  # 只测5个节省输出
        mock_expert_team
    )
    
    # 评测 TrainingPlan
    print("\n📋 评测 TrainingPlan...")
    plan_result = evaluator.evaluate_module(
        "training_plan",
        TEST_CASES["training_plan"][:5],
        mock_training_plan
    )
    
    # 生成报告
    evaluator.generate_report("evaluation/reports/local_evaluation_report.md")
    
    print("\n" + "="*70)
    print(" 评测结果:")
    print(f" ExpertTeam: {expert_result['total_score']}/10.0 ({expert_result['grade']})")
    print(f" TrainingPlan: {plan_result['total_score']}/10.0 ({plan_result['grade']})")
    print("="*70)


if __name__ == "__main__":
    run_local_evaluation()
