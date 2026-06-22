"""
RAGAS 长期记忆系统评测

评测场景：用户问"我之前说过什么"时，系统能否准确召回相关记忆

评测指标：
- Context Precision: 检索记忆的相关性
- Context Recall: 是否漏掉重要记忆
- Faithfulness: AI 是否基于记忆回答

运行方式:
    python -m evaluation.ragas_memory_evaluator
"""

import os
import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import httpx

# 导入 RAGAS
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    RAGAS_AVAILABLE = True
except ImportError:
    print("警告: ragas 未安装")
    RAGAS_AVAILABLE = False


@dataclass
class MemoryTestCase:
    """记忆系统测试用例"""
    user_id: str
    question: str  # 比如"我之前有什么口吃症状？"
    ground_truth_memories: List[str]  # 应该被检索到的记忆内容
    expected_answer: str  # 期望的回答


class MemoryRAGASEvaluator:
    """长期记忆系统 RAGAS 评测器"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url.rstrip("/")
        
    async def save_test_memories(self, user_id: str, memories: List[Dict]):
        """保存测试记忆到系统"""
        async with httpx.AsyncClient() as client:
            for memory in memories:
                try:
                    await client.post(
                        f"{self.base_url}/memory/save",
                        json={
                            "user_id": user_id,
                            "dialogue": memory["content"],
                            "topic": memory.get("topic", "general"),
                            "timestamp": memory.get("timestamp", datetime.now().isoformat())
                        },
                        timeout=10.0
                    )
                except Exception as e:
                    print(f"保存记忆失败: {e}")
            print(f"✅ 已为用户 {user_id} 保存 {len(memories)} 条测试记忆")
    
    async def query_memory_system(self, user_id: str, question: str, go_api_url: str = None, token: str = None) -> Dict[str, Any]:
        """
        调用 Go 后端的真实业务接口进行 RAGAS 评测
        
        这个接口使用真实的业务逻辑：
        1. 检索长期记忆
        2. 使用真实的提示词模板
        3. 调用豆包 API 生成回答
        
        返回: {
            "question": str,
            "retrieved_memories": List[str],  # 检索到的记忆
            "answer": str  # AI 基于记忆的回复
        }
        """
        # 优先使用 Go API
        if go_api_url:
            return await self._call_go_evaluate_api(user_id, question, go_api_url, token)
        
        # 降级：使用 Python 本地接口（仅用于测试）
        return await self._call_python_local_api(user_id, question)
    
    async def _call_go_evaluate_api(self, user_id: str, question: str, go_api_url: str, token: str = None) -> Dict[str, Any]:
        """调用 Go 后端的 /evaluate-rag 接口（真实业务逻辑）"""
        async with httpx.AsyncClient() as client:
            try:
                headers = {}
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                
                response = await client.post(
                    f"{go_api_url}/api/v1/evaluation/rag",
                    headers=headers,
                    json={
                        "user_id": user_id,
                        "question": question
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                resp_data = response.json()
                
                # Go 接口返回格式: {"code": 0, "message": "...", "data": {...}}
                if resp_data.get("code") != 0:
                    raise Exception(f"Go API 错误: {resp_data.get('message', 'unknown')}")
                
                data = resp_data.get("data", {})
                
                return {
                    "question": question,
                    "retrieved_memories": data.get("contexts", []),
                    "answer": data.get("answer", ""),
                    "context_ids": data.get("context_ids", []),
                    "model_used": data.get("model_used", "unknown"),
                    "timing": {
                        "retrieval_ms": data.get("retrieval_time_ms", 0),
                        "generate_ms": data.get("generate_time_ms", 0)
                    }
                }
                
            except Exception as e:
                print(f"Go API 调用失败: {e}")
                return {
                    "question": question,
                    "retrieved_memories": [],
                    "answer": f"错误: {str(e)}"
                }
    
    async def _call_python_local_api(self, user_id: str, question: str) -> Dict[str, Any]:
        """调用 Python 本地接口（降级方案）"""
        async with httpx.AsyncClient() as client:
            try:
                # 1. 检索记忆
                search_response = await client.post(
                    f"{self.base_url}/memory/search",
                    json={
                        "user_id": user_id,
                        "query": question,
                        "n_results": 5
                    },
                    timeout=10.0
                )
                search_data = search_response.json()
                
                # 提取检索到的记忆内容
                retrieved_memories = []
                for result in search_data.get("results", []):
                    if result.get("dialogue"):
                        retrieved_memories.append(result["dialogue"])
                
                # 2. 基于记忆生成回答
                answer = self.call_doubao_for_memory(
                    question=question,
                    memories=retrieved_memories,
                    prompt_version="v1_基础"
                )
                
                return {
                    "question": question,
                    "retrieved_memories": retrieved_memories,
                    "answer": answer
                }
                
            except Exception as e:
                print(f"本地查询失败: {e}")
                return {
                    "question": question,
                    "retrieved_memories": [],
                    "answer": f"错误: {str(e)}"
                }
    
    # ==================== 提示词 A/B 测试 ====================
    
    PROMPT_VERSIONS = {
        "v1_基础": """你是 Fluent Life 口吃矫正助手的记忆回顾功能。

用户的长期记忆记录如下：
{memories}

你的任务：
1. 基于上述记忆回答用户的问题
2. 如果记忆中找不到相关信息，请诚实说明
3. 保持温暖、专业的语气
4. 不要编造记忆中不存在的信息
""",
        "v2_强调约束": """你是 Fluent Life 口吃矫正助手的记忆回顾功能。

⚠️ 重要约束：
- 你必须严格基于以下记忆内容回答
- 禁止添加记忆中不存在的信息
- 如果记忆不相关，必须说"根据您的记录，我没有找到相关信息"

用户的长期记忆记录如下：
{memories}

回答要求：简洁、准确、只提及记忆中的事实。
""",
        "v3_示例引导": """你是 Fluent Life 口吃矫正助手的记忆回顾功能。

用户的长期记忆记录如下：
{memories}

回答规则：
✓ 只能使用记忆中明确提到的信息
✓ 不要推测、不要补充背景知识
✓ 不确定时直接承认"根据您的记录，我没有相关信息"

示例：
- 记忆："用户有首字重复症状"
- 好回答："根据您的记录，您有首字重复的症状。"
- 坏回答："首字重复是心理障碍导致的，建议您..."（多了推测）
"""
    }
    
    def call_doubao_for_memory(self, question: str, memories: List[str], prompt_version: str = "v1_基础") -> str:
        """
        调用豆包 AI 基于记忆生成回答
        支持不同提示词版本进行 A/B 测试
        
        Args:
            prompt_version: 使用哪个提示词版本 (v1_基础/v2_强调约束/v3_示例引导)
        """
        from openai import OpenAI
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        
        model = os.getenv("DOUBAO_MODEL_ID", "ep-m-20260113142855-wqkg9")
        api_key = os.getenv("DOUBAO_API_KEY", "")
        
        # 如果没有 API Key，使用 Mock
        if not api_key:
            print("⚠️ 未配置 DOUBAO_API_KEY，使用 Mock 模式")
            if not memories:
                return "根据您的历史记录，我没有找到相关信息。"
            context = "\n".join([f"- {m}" for m in memories[:3]])
            return f"【Mock回复-{prompt_version}】根据您之前的记录：\n{context}\n\n关于您的问题，建议咨询专业医生。"
        
        # 构建记忆文本
        memories_text = ""
        if memories:
            for i, memory in enumerate(memories, 1):
                memories_text += f"\n[{i}] {memory}"
        else:
            memories_text = "\n（暂无相关记忆记录）"
        
        # 获取指定版本的提示词
        prompt_template = self.PROMPT_VERSIONS.get(prompt_version, self.PROMPT_VERSIONS["v1_基础"])
        system_prompt = prompt_template.format(memories=memories_text)
        
        # 调用豆包
        client = OpenAI(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"豆包调用失败: {e}")
            return f"抱歉，生成回答时出现错误: {str(e)}"
    
    def evaluate_with_ragas(self, test_results: List[Dict]) -> Dict[str, Any]:
        """使用 RAGAS 评测 - 配置豆包模型"""
        if not RAGAS_AVAILABLE:
            return {"error": "RAGAS 未安装"}
        
        # 准备数据 - 确保 contexts 是 Sequence[string] 类型
        from datasets import Dataset, Features, Sequence, Value
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
        from langchain_core.embeddings import Embeddings
        
        # 过滤空结果
        valid_results = [r for r in test_results if r.get("retrieved_memories")]
        if not valid_results:
            print("⚠️ 没有有效的检索结果，跳过 RAGAS 评测")
            return {}
        
        data = {
            "question": [r["question"] for r in valid_results],
            "contexts": [list(r["retrieved_memories"]) for r in valid_results],
            "answer": [r["answer"] for r in valid_results],
            "ground_truth": [r.get("expected_answer", "") for r in valid_results]
        }
        
        # 显式指定特征类型
        features = Features({
            "question": Value("string"),
            "contexts": Sequence(Value("string")),
            "answer": Value("string"),
            "ground_truth": Value("string")
        })
        
        dataset = Dataset.from_dict(data, features=features)
        
        # 配置豆包模型
        DOUBAO_KEY = os.getenv("DOUBAO_API_KEY") or os.popen("grep DOUBAO_API_KEY .env 2>/dev/null | cut -d= -f2").read().strip()
        
        doubao_llm = ChatOpenAI(
            model="ep-m-20260113142855-wqkg9",
            api_key=DOUBAO_KEY,
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            temperature=0.3
        )
        
        # Mock 嵌入
        class MockEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[0.1] * 1536 for _ in texts]
            def embed_query(self, text):
                return [0.1] * 1536
        
        ragas_llm = LangchainLLMWrapper(doubao_llm)
        
        # 评测
        results = evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall, faithfulness],
            llm=ragas_llm,
            embeddings=MockEmbeddings(),
            raise_exceptions=False
        )
        
        return results
    
    async def run_evaluation(
        self, 
        go_api_url: str = None, 
        token: str = None,
        use_go_api: bool = True
    ) -> Dict[str, Any]:
        """
        运行完整 RAGAS 评测
        
        Args:
            go_api_url: Go 后端 API 地址（如 http://localhost:8080）
            token: JWT 认证令牌（如果需要）
            use_go_api: 是否使用 Go API（True=真实业务逻辑，False=本地模拟）
        """
        print("\n" + "="*60)
        print("长期记忆系统 RAGAS 评测")
        if use_go_api and go_api_url:
            print(f"模式: 真实业务接口 ({go_api_url})")
        else:
            print("模式: 本地模拟")
        print("="*60)
        
        # 准备测试数据
        test_user_id = "test_user_ragas_001"
        
        # 1. 测试记忆（Go 端已硬编码，无需保存）
        print(f"✅ 测试用户 {test_user_id} 记忆已在 Go 端配置")
        
        # 2. 准备测试问题
        test_cases = [
            {
                "question": "我之前有什么口吃症状？",
                "expected_memories": ["用户自述：我说话时第一个字总是重复"],
                "expected_answer": "根据您之前的记录，您有首字重复的症状"
            },
            {
                "question": "我在什么场景下容易口吃？",
                "expected_memories": ["在打电话时特别紧张"],
                "expected_answer": "您提到在打电话时容易口吃"
            },
            {
                "question": "我之前试过什么治疗方法？",
                "expected_memories": ["尝试过节拍器训练"],
                "expected_answer": "您之前尝试过节拍器训练"
            }
        ]
        
        # 3. 运行测试
        results = []
        for tc in test_cases:
            print(f"\n📋 测试: {tc['question']}")
            
            # 调用评测接口
            if use_go_api and go_api_url:
                result = await self.query_memory_system(
                    test_user_id, 
                    tc["question"],
                    go_api_url=go_api_url,
                    token=token
                )
            else:
                result = await self.query_memory_system(test_user_id, tc["question"])
            
            result["expected_answer"] = tc["expected_answer"]
            result["expected_memories"] = tc["expected_memories"]
            results.append(result)
            
            print(f"   检索到 {len(result['retrieved_memories'])} 条记忆")
            print(f"   回答: {result['answer'][:80]}...")
            if result.get('timing'):
                print(f"   耗时: {result['timing']}")
        
        # 4. RAGAS 评测
        ragas_results = None
        if RAGAS_AVAILABLE:
            print("\n🔍 计算 RAGAS 指标...")
            ragas_results = self.evaluate_with_ragas(results)
            
            print("\n" + "="*60)
            print("评测结果")
            print("="*60)
            for metric, scores in ragas_results.items():
                if isinstance(scores, list):
                    avg_score = sum(scores) / len(scores) if scores else 0
                else:
                    avg_score = scores  # 可能是单个 float 值
                print(f"{metric}: {avg_score:.3f}")
        else:
            print("\n⚠️ RAGAS 未安装，跳过指标计算")
            print("   安装: pip install ragas")
        
        return {
            "test_cases": len(results),
            "details": results,
            "ragas_scores": ragas_results
        }


async def main():
    """主函数"""
    import os
    
    # 配置 RAGAS 使用豆包 API
    DOUBAO_KEY = os.getenv("DOUBAO_API_KEY") or os.popen("grep DOUBAO_API_KEY .env 2>/dev/null | cut -d= -f2").read().strip()
    if DOUBAO_KEY:
        os.environ["OPENAI_API_KEY"] = DOUBAO_KEY
        os.environ["OPENAI_BASE_URL"] = "https://ark.cn-beijing.volces.com/api/v3"
        print("✅ RAGAS 已配置豆包 API")
    
    # 从环境变量读取配置
    go_api_url = os.getenv("GO_API_URL", "http://localhost:8001")
    token = os.getenv("GO_API_TOKEN", "")
    use_go_api = os.getenv("USE_GO_API", "true").lower() == "true"
    
    evaluator = MemoryRAGASEvaluator()
    results = await evaluator.run_evaluation(
        go_api_url=go_api_url if use_go_api else None,
        token=token if token else None,
        use_go_api=use_go_api
    )
    
    # 保存报告
    import os
    from datetime import datetime
    os.makedirs("evaluation/reports", exist_ok=True)
    report_path = "evaluation/reports/ragas_memory_report.json"
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_cases": results['test_cases'],
        "ragas_scores": results['ragas_scores'],
        "details": results['details']
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 打印摘要
    print("\n" + "="*60)
    print("评测完成!")
    print(f"测试用例数: {results['test_cases']}")
    if results['ragas_scores']:
        print("RAGAS 指标已计算")
    print(f"报告已保存: {report_path}")
    print("="*60)


# 使用说明:
# 
# 1. 使用 Go 后端真实业务逻辑（推荐）:
#    export GO_API_URL=http://localhost:8080
#    export GO_API_TOKEN=your_jwt_token  # 如果需要认证
#    export USE_GO_API=true
#    python -m evaluation.ragas_memory_evaluator
#
# 2. 使用本地模拟（测试用）:
#    export USE_GO_API=false
#    python -m evaluation.ragas_memory_evaluator


if __name__ == "__main__":
    asyncio.run(main())
