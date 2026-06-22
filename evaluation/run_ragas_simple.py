"""
简化版 RAGAS 评估 - 直接调用豆包 API 进行评测
绕过复杂的依赖问题
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

# 配置豆包 API（RAGAS 使用 OpenAI 兼容模式）
DOUBAO_KEY = os.getenv("DOUBAO_API_KEY") or os.popen("grep DOUBAO_API_KEY .env 2>/dev/null | cut -d= -f2").read().strip()
if DOUBAO_KEY:
    os.environ["DOUBAO_API_KEY"] = DOUBAO_KEY  # 确保 call_doubao 能读取到
    os.environ["OPENAI_API_KEY"] = DOUBAO_KEY
    os.environ["OPENAI_BASE_URL"] = "https://ark.cn-beijing.volces.com/api/v3"
    print(f"✅ 已配置豆包 API")
else:
    print("⚠️ 未找到 DOUBAO_API_KEY")

# 导入 RAGAS
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
    print("✅ RAGAS 已加载")
except ImportError as e:
    print(f"❌ RAGAS 导入失败: {e}")
    RAGAS_AVAILABLE = False


@dataclass
class RAGTestCase:
    """RAG 测试用例"""
    question: str
    ground_truth: Optional[str] = None
    contexts: List[str] = None  # 预设的上下文
    
    def __post_init__(self):
        if self.contexts is None:
            self.contexts = []


# 测试用例 - 口吃矫正领域
RAG_TEST_CASES = [
    RAGTestCase(
        question="什么是口吃？",
        ground_truth="口吃是一种言语流畅性障碍，表现为说话时出现重复、延长或阻塞等症状。",
        contexts=[
            "口吃（Stuttering）是一种言语流畅性障碍，主要表现为说话时出现不自主的重复、延长或阻塞。",
            "口吃通常在儿童早期开始，男性发病率高于女性。",
            "口吃的成因包括遗传因素、神经生理因素和心理因素等。"
        ]
    ),
    RAGTestCase(
        question="首字难发型口吃有什么特点？",
        ground_truth="首字难发型口吃表现为说话时在第一个字或音节上出现重复、延长或阻塞，难以发出第一个音。",
        contexts=[
            "首字难发型是最常见的口吃类型之一，患者在发第一个字时会遇到困难。",
            "表现为第一个字重复多次，如'我我我'，或者第一个音延长，如'www我'。",
            "通常在句子开头或重要词汇的首字上表现最明显。"
        ]
    ),
    RAGTestCase(
        question="呼吸训练对口吃矫正有帮助吗？",
        ground_truth="呼吸训练有助于口吃矫正，通过调整呼吸节奏和气流控制，可以帮助说话者放松，减少言语紧张。",
        contexts=[
            "呼吸训练是口吃矫正的基础方法之一，通过腹式呼吸练习来调节气息。",
            "正确的呼吸方式可以帮助说话者放松，减少说话时的紧张感。",
            "建议在说话前先吸气，保持气息平稳，避免因气息不足导致的言语中断。"
        ]
    ),
    RAGTestCase(
        question="儿童口吃的最佳治疗年龄是多大？",
        contexts=[
            "儿童口吃的最佳干预期通常在2-6岁之间，这个阶段语言发展迅速。",
            "早期干预效果更好，可以避免口吃成为长期习惯。",
            "家长应保持耐心，不要过度纠正孩子的说话方式，避免造成心理压力。"
        ]
    ),
    RAGTestCase(
        question="口吃让我变得很自卑，不敢和人交流怎么办？",
        contexts=[
            "口吃可能带来心理负担，导致社交焦虑和自卑情绪。",
            "建议寻求心理咨询帮助，建立积极的自我认知。",
            "加入口吃互助小组，与有相似经历的人交流可以获得支持和鼓励。",
            "渐进式暴露疗法可以帮助逐步克服社交恐惧。"
        ]
    ),
]


class SimpleRAGASEvaluator:
    """简化版 RAGAS 评测器"""
    
    def __init__(self):
        self.results = []
        
    def call_doubao(self, question: str, contexts: List[str]) -> str:
        """调用豆包 API 生成回答"""
        from openai import OpenAI
        
        api_key = os.getenv("DOUBAO_API_KEY")
        model = os.getenv("DOUBAO_MODEL_ID", "ep-m-20260113142855-wqkg9")
        
        if not api_key:
            print("⚠️ 未配置 DOUBAO_API_KEY，使用 Mock 回答")
            return f"【Mock】基于提供的上下文，我可以回答：{question[:20]}..."
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
        
        context_text = "\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": f"你是 Fluent Life 口吃矫正助手。基于以下参考资料回答用户问题：\n\n{context_text}\n\n要求：\n1. 只基于参考资料回答\n2. 保持简洁专业\n3. 如果不确定，诚实说明"
                    },
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ 豆包调用失败: {e}")
            return f"错误: {str(e)}"
    
    async def evaluate_single(self, test_case: RAGTestCase) -> Dict:
        """评测单个用例"""
        print(f"\n📋 问题: {test_case.question}")
        
        # 调用豆包生成回答
        answer = self.call_doubao(test_case.question, test_case.contexts)
        print(f"   回答: {answer[:80]}...")
        
        return {
            "question": test_case.question,
            "contexts": test_case.contexts,
            "answer": answer,
            "ground_truth": test_case.ground_truth or ""
        }
    
    def calculate_ragas(self, results: List[Dict]) -> Dict:
        """计算 RAGAS 指标 - 使用豆包模型"""
        if not RAGAS_AVAILABLE:
            return {"error": "RAGAS 不可用"}
        
        # 准备数据集
        data = {
            "question": [r["question"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "answer": [r["answer"] for r in results],
            "ground_truth": [r["ground_truth"] for r in results]
        }
        
        dataset = Dataset.from_dict(data)
        
        print("\n🔍 计算 RAGAS 指标中...")
        
        try:
            # 配置豆包模型（OpenAI 兼容模式）
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            
            api_key = os.getenv("DOUBAO_API_KEY") or os.popen("grep DOUBAO_API_KEY .env 2>/dev/null | cut -d= -f2").read().strip()
            
            # 初始化豆包模型
            doubao_llm = ChatOpenAI(
                model="ep-m-20260113142855-wqkg9",
                api_key=api_key,
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                temperature=0.3
            )
            
            # 使用简单的嵌入（豆包没有 embedding API，用 mock）
            from langchain_core.embeddings import Embeddings
            class MockEmbeddings(Embeddings):
                def embed_documents(self, texts):
                    return [[0.1] * 1536 for _ in texts]
                def embed_query(self, text):
                    return [0.1] * 1536
            
            mock_embeddings = MockEmbeddings()
            
            # 包装为 RAGAS 格式
            ragas_llm = LangchainLLMWrapper(doubao_llm)
            ragas_embeddings = LangchainEmbeddingsWrapper(mock_embeddings)
            
            # 选择指标
            metrics = [
                faithfulness,
                answer_relevancy, 
                context_precision,
                context_recall
            ]
            
            # 只有有 ground_truth 的才评测 correctness
            has_ground_truth = any(r["ground_truth"] for r in results)
            if has_ground_truth:
                metrics.append(answer_correctness)
            
            # 运行评测（传入自定义模型）
            scores = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                raise_exceptions=False
            )
            
            return scores
            
        except Exception as e:
            print(f"❌ RAGAS 计算失败: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    async def run_evaluation(self):
        """运行完整评测"""
        print("="*60)
        print("🚀 Fluent Life RAGAS 评测 (简化版)")
        print("="*60)
        print(f"\n📊 测试用例数: {len(RAG_TEST_CASES)}")
        print(f"RAGAS 可用: {RAGAS_AVAILABLE}")
        
        # 1. 逐个评测
        results = []
        for tc in RAG_TEST_CASES:
            result = await self.evaluate_single(tc)
            results.append(result)
            await asyncio.sleep(0.5)  # 避免 API 限流
        
        # 2. 计算 RAGAS
        ragas_scores = None
        if RAGAS_AVAILABLE:
            ragas_scores = self.calculate_ragas(results)
            
            # 3. 输出结果
            print("\n" + "="*60)
            print("📈 RAGAS 评测结果")
            print("="*60)
            
            if "error" in ragas_scores:
                print(f"❌ 评测失败: {ragas_scores['error']}")
            else:
                # 打印各项指标的均值
                for metric, values in ragas_scores.items():
                    if isinstance(values, list) and len(values) > 0:
                        avg = sum(values) / len(values)
                        print(f"  {metric}: {avg:.3f}")
                    else:
                        print(f"  {metric}: {values}")
        else:
            print("\n⚠️ RAGAS 未安装，跳过指标计算")
        
        # 4. 保存报告
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_cases": len(results),
            "ragas_scores": {
                k: (sum(v)/len(v) if isinstance(v, list) and len(v) > 0 else v) 
                for k, v in (ragas_scores or {}).items() 
                if k != 'error'
            } if ragas_scores else None,
            "results": results
        }
        
        output_path = "evaluation/reports/ragas_simple_report.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 报告已保存: {output_path}")


async def main():
    evaluator = SimpleRAGASEvaluator()
    await evaluator.run_evaluation()


if __name__ == "__main__":
    asyncio.run(main())
