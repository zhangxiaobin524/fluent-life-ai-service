# RAGAS 评估系统完全指南

## 目录
1. [什么是 RAGAS](#什么是-ragas)
2. [核心评估指标](#核心评估指标)
3. [项目架构与流程](#项目架构与流程)
4. [代码详解](#代码详解)
5. [如何运行评估](#如何运行评估)
6. [结果解读与优化](#结果解读与优化)

---

## 什么是 RAGAS

### 概念
**RAGAS** (Retrieval-Augmented Generation Assessment) 是一个开源框架，用于**自动评估 RAG (检索增强生成) 系统**的质量。

### 为什么需要 RAGAS？
传统的 RAG 系统评估需要人工标注，成本高、效率低。RAGAS 通过**LLM-as-Judge** 的方式，让大模型自动评估回答质量。

### 核心思想
```
┌─────────────────────────────────────────────────────────┐
│  RAGAS 评估流程                                          │
├─────────────────────────────────────────────────────────┤
│  1. 提出问题 (Question)                                  │
│  2. 系统检索相关文档 (Contexts)                          │
│  3. 生成回答 (Answer)                                    │
│  4. RAGAS 使用 LLM 评估回答质量                          │
└─────────────────────────────────────────────────────────┘
```

---

## 核心评估指标

### 📌 三句话总结

> **Precision（精确率）**: 我拿到的资料里，有多少是相关的？→ 衡量「准不准」
> **Recall（召回率）**: 所有相关资料里，我找到了多少？→ 衡量「全不全」
> **Faithfulness（忠实度）**: AI 有没有「胡说八道」？→ 衡量「诚不诚实」

用图书馆找书的比喻：
- **Precision 高** = 借回来的书都有用，没有一本白借 ✅
- **Recall 高** = 所有有用的书都找到了，没有遗漏 ✅
- **Faithfulness 高** = AI 老老实实基于找到的书回答，不瞎编 ✅

---

### 1. Context Precision (上下文精确率)
**定义**: 检索到的文档中有多少是真正相关的

**通俗理解**:
> 想象你去图书馆找书，Precision 衡量的是：你借回来的书里，有多少是真正对你有用的。
> - Precision = 1.0: 借5本，5本都有用，没有一本是白借的 👍
> - Precision = 0.2: 借5本，只有1本有用，浪费了4本 😅

**计算公式**:
```
Context Precision = 相关文档数 / 检索到的总文档数
```

**示例**:
```
问题: "我之前有什么口吃症状？"
检索结果:
✅ "用户自述：我说话时第一个字总是重复..."  (相关)
✅ "用户提到：在打电话时特别紧张..."       (相关)
❌ "用户喜欢运动..."                        (不相关)

Context Precision = 2/3 = 0.667
```

**实际意义**: 0.289 表示系统检索到的记忆中，约 **71% 是不相关的**。用户问的是口吃症状，但系统可能返回了用户的饮食习惯、运动爱好等无关信息。这会导致 AI 回答质量下降。

### 2. Context Recall (上下文召回率)
**定义**: 所有相关文档中被成功检索到的比例

**通俗理解**:
> 继续用图书馆的例子，Recall 衡量的是：图书馆里所有对你有用的书，你找到了多少。
> - Recall = 1.0: 图书馆有10本对你有用的书，你全找到了，没有遗漏 👍
> - Recall = 0.3: 图书馆有10本对你有用的书，你只找到3本，漏了7本 😅
>
> **注意**: Precision 和 Recall 往往是此消彼长的。严格挑选（Precision高）可能漏掉一些有用的（Recall低）；宁滥勿缺（Recall高）可能带回来很多没用的（Precision低）。

**计算公式**:
```
Context Recall = 检索到的相关文档数 / 所有相关文档数
```

**示例**:
```
所有相关记忆: ["首字重复", "打电话紧张", "尝试节拍器"]
检索到的记忆: ["首字重复", "打电话紧张"]

Context Recall = 2/3 = 0.667
```

**实际意义**: 0.667 表示系统成功找回了约 **67% 的相关记忆**。比如用户有3条关于口吃的记录，系统找到了其中2条，漏掉了1条。漏掉的可能是关键信息，导致 AI 回答不全面。

### 3. Faithfulness (忠实度)
**定义**: AI 回答是否基于检索到的文档，有没有"幻觉"

**通俗理解**:
> 想象你是老师，学生考试时能不能"照本宣科"。Faithfulness 衡量的是：AI 的回答里，有多少内容是可以从给定的资料里找到依据的。
> - Faithfulness = 1.0: AI 完全基于提供的记忆回答，没有添油加醋 👍
> - Faithfulness = 0.3: AI 的回答里只有 30% 是基于记忆的，70% 是"自由发挥"（幻觉）⚠️
>
> **LLM 幻觉**: 大模型有时候会"一本正经地胡说八道"，比如用户的记忆里只说"我有口吃"，AI 却回答"根据您的记录，您的口吃是由童年创伤引起的"——这就是幻觉，Faithfulness 低。

**计算公式**:
```
Faithfulness = 回答中可从文档验证的陈述数 / 回答中总陈述数
```

**示例**:
```
检索内容: "用户有首字重复症状，持续5年"

回答A: "根据您的记录，您有首字重复的症状。" → Faithfulness = 1.0 ✅
回答B: "首字重复是心理障碍导致的，建议您..." → Faithfulness = 0.5 ⚠️ (后半句无依据)
```

**实际意义**: 0.389 表示 AI 回答中约 **61% 的内容是"编造的"或"无法验证的"**。这在医疗/健康场景很危险——AI 可能会给用户错误的建议。需要优化提示词，明确告诉 AI "只能基于提供的记忆回答，不能推测"。

### 4. Answer Relevancy (回答相关性)
**定义**: 回答是否与问题相关

---

## 项目架构与流程

### 系统架构
```
┌────────────────────────────────────────────────────────────────┐
│                        RAGAS 评估架构                          │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │  Python      │────▶│  Go 后端      │────▶│  豆包 API    │   │
│  │  评估脚本     │     │  (Port 8001) │     │  (LLM)       │   │
│  └──────────────┘     └──────┬───────┘     └──────────────┘   │
│         │                    │                                 │
│         │                    ▼                                 │
│         │            ┌──────────────┐                         │
│         │            │  Python 向量库│                         │
│         └───────────▶│  (Port 8002) │                         │
│                      │  ChromaDB    │                         │
│                      └──────────────┘                         │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 数据流转
```
1. 评估脚本 (Python) 发送请求到 Go 后端
   POST /api/v1/evaluation/rag
   {
     "user_id": "test_user_ragas_001",
     "question": "我之前有什么口吃症状？"
   }

2. Go 后端调用向量库服务检索记忆
   POST /memory/search
   → 返回相关记忆列表

3. Go 后端调用豆包 API 生成回答
   → 基于检索到的记忆生成答案

4. Go 后端返回完整结果给评估脚本
   {
     "question": "...",
     "contexts": [...],     // 检索到的记忆
     "answer": "...",       // AI 回答
     "timing": {...}        // 耗时统计
   }

5. Python 评估脚本调用 RAGAS 计算指标
   → 使用豆包作为 Judge LLM
   → 输出 Context Precision / Recall / Faithfulness
```

---

## 代码详解

### 1. Go 后端 - 评估接口

**文件**: `fluent-life-api/internal/handlers/ai_handler.go`

```go
// EvaluateRAG 长期记忆系统RAGAS评测接口
func (h *AIHandler) EvaluateRAG(c *gin.Context) {
    var req RAGASEvaluationRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        response.BadRequest(c, "请求参数错误")
        return
    }

    // 1. 检索长期记忆（带时间统计）
    retrievalStart := time.Now()
    memories, searchErr := h.aiService.SearchLongTermMemory(userID, req.Question)
    retrievalTime := float64(time.Since(retrievalStart).Milliseconds())
    
    // 2. 如果是测试用户且向量库为空，写入测试数据
    if req.UserID == "test_user_ragas_001" && (searchErr != nil || len(memories) == 0) {
        // 写入测试记忆到向量库
        for _, mem := range testMemories {
            http.Post(pythonURL+"/memory/save", ...)
        }
        time.Sleep(500 * time.Millisecond) // 等待写入
        memories, _ = h.aiService.SearchLongTermMemory(userID, req.Question)
    }

    // 3. 构建系统提示词（复用业务逻辑）
    systemPrompt := "你是 Fluent Life 口吃矫正AI导师。"
    if len(memories) > 0 {
        systemPrompt += "\n\n【用户历史相关对话】\n"
        for i, mem := range memories {
            if i >= 2 { break }  // 最多使用2条
            systemPrompt += fmt.Sprintf("%d. %s\n", i+1, mem)
        }
    }

    // 4. 调用豆包生成回答
    generateStart := time.Now()
    messages := []models.Message{
        {Role: "system", Text: systemPrompt},
        {Role: "user", Text: req.Question},
    }
    answer, _ := h.aiService.CallDoubaoAPIForEvaluation(messages)
    generateTime := float64(time.Since(generateStart).Milliseconds())

    // 5. 返回RAGAS需要的数据
    evalResp := RAGASEvaluationResponse{
        Question:      req.Question,
        Contexts:      memories,
        Answer:        answer,
        RetrievalTime: retrievalTime,
        GenerateTime:  generateTime,
    }
    response.Success(c, evalResp, "RAG评测完成")
}
```

### 2. Python 向量库服务

**文件**: `fluent-life-ai-service/memory_service_only.py`

```python
@app.post("/memory/save")
async def save_long_term_memory(req: LongTermMemoryRequest):
    """保存对话到长期记忆"""
    # 获取或创建集合（每个用户一个集合）
    collection_name = f"memories_{req.user_id}"
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # 生成 embedding
    embedding = get_embedding(req.dialogue)
    
    # 添加到向量库
    collection.add(
        ids=[memory_id],
        embeddings=[embedding],
        documents=[req.dialogue],
        metadatas=[{"topic": req.topic, "timestamp": req.timestamp}]
    )
    return {"status": "success", "memory_id": memory_id}

@app.post("/memory/search")
async def search_long_term_memory(req: SearchLongTermMemoryRequest):
    """搜索长期记忆"""
    collection_name = f"memories_{req.user_id}"
    collection = chroma_client.get_collection(collection_name)
    
    # 生成查询 embedding
    query_embedding = get_embedding(req.query)
    
    # 向量相似度搜索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=req.n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    # 返回结果（兼容 Go 端格式）
    return {
        "results": memories,      # 详细结果
        "documents": documents,   # 纯文本列表（Go端需要）
        "count": len(memories)
    }
```

### 3. Python 评估脚本

**文件**: `fluent-life-ai-service/evaluation/ragas_memory_evaluator.py`

```python
class MemoryRAGASEvaluator:
    async def query_memory_system(self, user_id: str, question: str, go_api_url: str) -> Dict:
        """调用 Go 后端的真实业务接口"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{go_api_url}/api/v1/evaluation/rag",
                json={"user_id": user_id, "question": question},
                timeout=30.0
            )
            resp_data = response.json()
            data = resp_data.get("data", {})
            
            return {
                "question": question,
                "retrieved_memories": data.get("contexts", []),
                "answer": data.get("answer", ""),
            }

    def evaluate_with_ragas(self, test_results: List[Dict]) -> Dict[str, Any]:
        """使用 RAGAS 评测 - 配置豆包模型"""
        # 准备数据 - 确保 contexts 是 Sequence[string] 类型
        from datasets import Dataset, Features, Sequence, Value
        
        data = {
            "question": [r["question"] for r in test_results],
            "contexts": [list(r["retrieved_memories"]) for r in test_results],
            "answer": [r["answer"] for r in test_results],
            "ground_truth": [r.get("expected_answer", "") for r in test_results]
        }
        
        features = Features({
            "question": Value("string"),
            "contexts": Sequence(Value("string")),
            "answer": Value("string"),
            "ground_truth": Value("string")
        })
        dataset = Dataset.from_dict(data, features=features)
        
        # 配置豆包模型作为 Judge
        doubao_llm = ChatOpenAI(
            model="ep-m-20260113142855-wqkg9",
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
        
        # 运行评测
        results = evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall, faithfulness],
            llm=LangchainLLMWrapper(doubao_llm),
            embeddings=MockEmbeddings(),
        )
        return results
```

---

## 如何运行评估

### 方式 1: 一键脚本（推荐）

```bash
chmod +x /Users/zhangxiaobin/self-project/fluent-life-ai-service/run_ragas_eval.sh
/Users/zhangxiaobin/self-project/fluent-life-ai-service/run_ragas_eval.sh
```

### 方式 2: 手动运行

```bash
# 1. 启动 Python 向量库服务
cd /Users/zhangxiaobin/self-project/fluent-life-ai-service
python3 memory_service_only.py

# 2. 启动 Go 后端（新终端）
cd /Users/zhangxiaobin/self-project/fluent-life-api
export PYTHON_AI_SERVICE_URL=http://localhost:8002
go run ./cmd/server/main.go

# 3. 运行评估（新终端）
cd /Users/zhangxiaobin/self-project/fluent-life-ai-service
export GO_API_URL=http://localhost:8001
export USE_GO_API=true
python3 -m evaluation.ragas_memory_evaluator
```

### 方式 3: 直接运行（服务已启动）

```bash
cd /Users/zhangxiaobin/self-project/fluent-life-ai-service
export GO_API_URL=http://localhost:8001
export USE_GO_API=true
python3 -m evaluation.ragas_memory_evaluator
```

---

## 结果解读与优化

### 示例输出

```
============================================================
评测结果
============================================================
context_precision: 0.289  ← 检索精确率偏低
context_recall: 0.667     ← 召回率较好
faithfulness: 0.389       ← 忠实度一般

📋 测试: 我之前有什么口吃症状？
   检索到 5 条记忆
   回答: 你说话时第一个字总是重复，比如"我我我"...
   耗时: {'retrieval_ms': 24, 'generate_ms': 800}
```

### 结果解读

| 指标 | 分数 | 通俗解读 | 通俗比喻 | 优化方向 |
|------|------|----------|----------|----------|
| **Context Precision** | 0.289 | 借10本书，只有3本有用，7本白借了 | 📚 借书的「命中率」| 优化向量模型、调整相似度阈值 |
| **Context Recall** | 0.667 | 图书馆有9本有用的书，你找到了6本，漏了3本 | 📚 找书的「完整度」| 增加检索数量、优化 Embedding |
| **Faithfulness** | 0.389 | AI 的回答里，61% 是「编」的，只有39%有依据 | 📝 回答的「诚实度」| 优化提示词、增加约束 |

### 优化建议

#### 1. 提升 Context Precision
- **优化 Embedding 模型**: 使用更适合中文的模型（如 BAAI/bge-large-zh）
- **调整相似度阈值**: 过滤低相似度结果
- **重排序 (Rerank)**: 使用更精确的重排序模型

#### 2. 提升 Context Recall
- **增加检索数量**: 从 top-3 增加到 top-5 或 top-10
- **查询扩展**: 使用 LLM 扩展用户查询的多种表述
- **混合检索**: 结合向量检索 + 关键词检索

#### 3. 提升 Faithfulness
- **提示词优化**: 明确告诉 AI "只能基于提供的记忆回答"
- **后处理校验**: 使用 LLM 检查回答中是否有无法验证的内容
- **引用标注**: 让 AI 标注回答中的信息来源

---

## 总结

### RAGAS 的优势
1. **自动化**: 无需人工标注，降低评估成本
2. **全面性**: 从检索和生成两个维度评估
3. **可解释性**: 每个指标都有明确的含义和计算方式

### 在 Fluent Life 中的应用
- 评估长期记忆系统的检索质量
- 评估 AI 导师的回答是否基于用户历史
- 持续监控 RAG 系统性能，指导优化方向

### 扩展建议
1. 增加更多测试用例覆盖不同场景
2. 集成到 CI/CD 流程，每次代码变更自动评估
3. 对比不同 Embedding 模型和提示词的效果

---

**文档版本**: v1.0  
**最后更新**: 2025-05-27  
**作者**: CodeBuddy
