"""
RAGAS 评测器 - 评估 RAG 系统质量

评测指标：
- Faithfulness: 忠实度（回答是否基于上下文）
- Answer Relevancy: 回答相关性
- Context Precision: 上下文精确度
- Context Recall: 上下文召回率
- Answer Correctness: 回答正确性（需要 ground_truth）

运行方式:
    python -m evaluation.ragas_evaluator

需要先运行 main.py 服务，然后执行评测
"""

import os
import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

import httpx
from datasets import Dataset

# 导入 RAGAS 指标
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    print("警告: ragas 未安装，请先运行: pip install ragas==0.1.21")
    RAGAS_AVAILABLE = False


@dataclass
class RAGTestCase:
    """RAG 测试用例"""
    question: str
    ground_truth: Optional[str] = None  # 标准答案（可选）
    user_context: Optional[str] = None  # 用户背景
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "ground_truth": self.ground_truth or "",
            "user_context": self.user_context or ""
        }


@dataclass
class RAGEvaluationResult:
    """RAG 评测结果"""
    question: str
    contexts: List[str]
    answer: str
    ground_truth: Optional[str]
    # RAGAS 分数
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_correctness: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "contexts": self.contexts,
            "answer": self.answer,
            "ground_truth": self.ground_truth,
            "scores": {
                "faithfulness": round(self.faithfulness, 3),
                "answer_relevancy": round(self.answer_relevancy, 3),
                "context_precision": round(self.context_precision, 3),
                "context_recall": round(self.context_recall, 3),
                "answer_correctness": round(self.answer_correctness, 3) if self.ground_truth else None
            }
        }


class RAGASEvaluator:
    """RAGAS 评测器"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.results: List[RAGEvaluationResult] = []
        
    async def call_rag_api(self, question: str, user_context: Optional[str] = None) -> Dict[str, Any]:
        """调用 RAG 对话接口获取回答和上下文"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/rag/chat",
                    json={
                        "question": question,
                        "collection_name": "stutter_correction",
                        "n_results": 3,
                        "user_context": user_context
                    }
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print(f"  ❌ API 调用失败: {e}")
                return {
                    "question": question,
                    "contexts": [],
                    "answer": f"错误: {str(e)}",
                    "model_used": "error"
                }
    
    async def evaluate_single(self, test_case: RAGTestCase) -> RAGEvaluationResult:
        """评测单个测试用例"""
        print(f"\n  📋 问题: {test_case.question[:50]}...")
        
        # 1. 调用 RAG 系统获取回答
        rag_response = await self.call_rag_api(
            question=test_case.question,
            user_context=test_case.user_context
        )
        
        contexts = rag_response.get("contexts", [])
        answer = rag_response.get("answer", "")
        
        print(f"     检索到 {len(contexts)} 条上下文")
        print(f"     回答长度: {len(answer)} 字符")
        
        result = RAGEvaluationResult(
            question=test_case.question,
            contexts=contexts,
            answer=answer,
            ground_truth=test_case.ground_truth
        )
        
        return result
    
    def calculate_ragas_scores(self, results: List[RAGEvaluationResult]) -> List[RAGEvaluationResult]:
        """使用 RAGAS 计算各项指标"""
        if not RAGAS_AVAILABLE:
            print("RAGAS 不可用，跳过评分")
            return results
        
        # 准备数据集
        data = {
            "question": [r.question for r in results],
            "contexts": [r.contexts for r in results],
            "answer": [r.answer for r in results],
            "ground_truth": [r.ground_truth or "" for r in results]
        }
        
        # 过滤掉没有 ground_truth 的数据用于 correctness 评测
        has_ground_truth = [r.ground_truth is not None and r.ground_truth.strip() != "" for r in results]
        
        dataset = Dataset.from_dict(data)
        
        print("\n  🔍 正在计算 RAGAS 指标...")
        
        # 选择评测指标
        metrics = [faithfulness, answer_relevancy, context_precision]
        
        # 只有部分数据有 ground_truth 时才评测 correctness
        if any(has_ground_truth):
            metrics.append(answer_correctness)
        
        try:
            # 运行评测
            ragas_results = evaluate(
                dataset=dataset,
                metrics=metrics,
                raise_exceptions=False  # 遇到错误继续
            )
            
            # 更新结果
            for i, result in enumerate(results):
                result.faithfulness = ragas_results.get("faithfulness", [0]*len(results))[i] or 0
                result.answer_relevancy = ragas_results.get("answer_relevancy", [0]*len(results))[i] or 0
                result.context_precision = ragas_results.get("context_precision", [0]*len(results))[i] or 0
                
                if has_ground_truth[i]:
                    result.answer_correctness = ragas_results.get("answer_correctness", [0]*len(results))[i] or 0
            
        except Exception as e:
            print(f"  ⚠️ RAGAS 评测出错: {e}")
            print("  可能原因: API 调用失败或返回格式问题")
        
        return results
    
    async def evaluate_batch(self, test_cases: List[RAGTestCase]) -> Dict[str, Any]:
        """批量评测"""
        print(f"\n{'='*60}")
        print(f"开始 RAGAS 评测 - 共 {len(test_cases)} 个测试用例")
        print(f"{'='*60}")
        
        # 1. 并行调用所有测试用例
        tasks = [self.evaluate_single(tc) for tc in test_cases]
        results = await asyncio.gather(*tasks)
        
        # 2. 计算 RAGAS 分数
        results = self.calculate_ragas_scores(results)
        
        self.results = results
        
        # 3. 计算汇总统计
        summary = self._calculate_summary(results)
        
        return summary
    
    def _calculate_summary(self, results: List[RAGEvaluationResult]) -> Dict[str, Any]:
        """计算汇总统计"""
        if not results:
            return {}
        
        # 各项指标平均分
        avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
        avg_relevancy = sum(r.answer_relevancy for r in results) / len(results)
        avg_precision = sum(r.context_precision for r in results) / len(results)
        
        # 有 ground_truth 的才计算 correctness
        correctness_scores = [r.answer_correctness for r in results if r.ground_truth]
        avg_correctness = sum(correctness_scores) / len(correctness_scores) if correctness_scores else None
        
        # 综合得分（加权平均）
        weights = {
            "faithfulness": 0.3,
            "answer_relevancy": 0.25,
            "context_precision": 0.25,
            "answer_correctness": 0.2
        }
        
        overall = (
            avg_faithfulness * weights["faithfulness"] +
            avg_relevancy * weights["answer_relevancy"] +
            avg_precision * weights["context_precision"]
        )
        
        if avg_correctness:
            overall += avg_correctness * weights["answer_correctness"]
        else:
            # 如果没有 correctness，重新归一化
            overall = overall / (weights["faithfulness"] + weights["answer_relevancy"] + weights["context_precision"])
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_cases": len(results),
            "cases_with_ground_truth": len(correctness_scores),
            "average_scores": {
                "faithfulness": round(avg_faithfulness, 3),
                "answer_relevancy": round(avg_relevancy, 3),
                "context_precision": round(avg_precision, 3),
                "answer_correctness": round(avg_correctness, 3) if avg_correctness else None
            },
            "overall_score": round(overall, 3),
            "grade": self._score_to_grade(overall),
            "details": [r.to_dict() for r in results]
        }
    
    def _score_to_grade(self, score: float) -> str:
        """分数转等级"""
        if score >= 0.9:
            return "A+ (优秀)"
        elif score >= 0.8:
            return "A (良好)"
        elif score >= 0.7:
            return "B (中等)"
        elif score >= 0.6:
            return "C (及格)"
        else:
            return "D (需改进)"
    
    def generate_report(self, output_path: str = "evaluation/ragas_report.json"):
        """生成评测报告"""
        summary = self._calculate_summary(self.results)
        
        # 保存 JSON 报告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 报告已保存: {output_path}")
        
        # 打印摘要
        print(f"\n{'='*60}")
        print("RAGAS 评测结果摘要")
        print(f"{'='*60}")
        print(f"综合得分: {summary.get('overall_score', 0)} ({summary.get('grade', 'N/A')})")
        print(f"评测用例: {summary.get('total_cases', 0)} 个")
        print(f"\n各项指标:")
        for metric, score in summary.get('average_scores', {}).items():
            if score is not None:
                print(f"  - {metric}: {score}")
        print(f"{'='*60}")
        
        return summary


# ==================== 测试用例 ====================

RAG_TEST_CASES: List[RAGTestCase] = [
    # 基础问答（无 ground_truth，评测忠实度和相关性）
    RAGTestCase(
        question="什么是口吃？",
        ground_truth="口吃是一种言语流畅性障碍，表现为说话时出现重复、延长或阻塞等症状。"
    ),
    RAGTestCase(
        question="首字难发型口吃有什么特点？",
        ground_truth="首字难发型口吃表现为说话时在第一个字或音节上出现重复、延长或阻塞，难以发出第一个音。"
    ),
    RAGTestCase(
        question="呼吸训练对口吃矫正有帮助吗？",
        ground_truth="呼吸训练有助于口吃矫正，通过调整呼吸节奏和气流控制，可以帮助说话者放松，减少言语紧张。"
    ),
    
    # 治疗相关问题
    RAGTestCase(
        question="流利说app的矫正方法是什么？",
        ground_truth="流利说app采用AI技术进行个性化训练，包括发音训练、流利度练习、心理调适等多种方法。"
    ),
    RAGTestCase(
        question="儿童口吃的最佳治疗年龄是多大？",
        user_context="用户有一个5岁的孩子"
    ),
    
    # 心理问题（无标准答案）
    RAGTestCase(
        question="口吃让我变得很自卑，不敢和人交流怎么办？",
        user_context="用户有社交焦虑"
    ),
    
    # 技巧咨询
    RAGTestCase(
        question="有什么简单的技巧可以减少口吃？",
        ground_truth="常见技巧包括：慢速说话、深呼吸、轻柔起音、停顿法、节奏控制等。"
    ),
    
    # 场景触发
    RAGTestCase(
        question="为什么我在打电话时更容易口吃？",
        ground_truth="电话交流时缺乏视觉反馈、时间压力大、对沉默的恐惧等因素会加重口吃。"
    ),
    
    # 长期管理
    RAGTestCase(
        question="口吃可以完全治愈吗？",
        ground_truth="口吃可以通过训练显著改善，但可能需要长期管理。早期干预效果更佳。"
    ),
    
    # 职场相关
    RAGTestCase(
        question="面试时如何克服口吃紧张？",
        ground_truth="面试前充分准备、练习自我介绍、使用慢速说话技巧、深呼吸放松等方法有帮助。"
    ),
]


async def main():
    """主函数"""
    print("\n🚀 RAGAS RAG 评测系统")
    print("确保 main.py 服务已启动 (python main.py)\n")
    
    # 等待服务启动
    await asyncio.sleep(1)
    
    # 创建评测器
    evaluator = RAGASEvaluator(base_url="http://localhost:8000")
    
    # 运行评测
    start_time = time.time()
    summary = await evaluator.evaluate_batch(RAG_TEST_CASES)
    elapsed = time.time() - start_time
    
    print(f"\n⏱️ 评测耗时: {elapsed:.1f} 秒")
    
    # 生成报告
    evaluator.generate_report("evaluation/reports/ragas_report.json")


if __name__ == "__main__":
    asyncio.run(main())
