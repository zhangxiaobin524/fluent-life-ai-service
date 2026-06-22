# RAGAS 长期记忆系统评测指南

## 架构说明

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Python 端      │     │    Go 后端        │     │    豆包 API      │
│  (RAGAS 评测)    │────▶│  (真实业务逻辑)   │────▶│   (LLM 生成)    │
│                 │     │                  │     │                 │
│ - 准备测试数据   │     │ - 检索长期记忆   │     │                 │
│ - 调用 Go 接口   │     │ - 组装提示词     │     │                 │
│ - 计算 RAGAS   │     │ - 调用豆包 API   │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

**为什么这样设计？**
- 评测必须基于**真实的业务逻辑**（提示词组装、记忆检索、LLM 调用）
- Go 后端负责实际的用户交互，Python 端只提供记忆存储/检索
- 这样可以 100% 反映生产环境的表现

## 启动步骤

### 1. 启动 Python 记忆服务

```bash
cd fluent-life-ai-service
python main.py
# 默认端口: 8000
```

### 2. 启动 Go 后端服务

```bash
cd fluent-life-api
go run cmd/server/main.go
# 默认端口: 8080
```

### 3. 运行 RAGAS 评测

```bash
cd fluent-life-ai-service

# 方式1: 使用 Go 后端真实业务逻辑（推荐）
export GO_API_URL=http://localhost:8080
export GO_API_TOKEN=your_jwt_token  # 如果需要认证
export USE_GO_API=true
python -m evaluation.ragas_memory_evaluator

# 方式2: 使用本地模拟（仅用于测试 Python 端）
export USE_GO_API=false
python -m evaluation.ragas_memory_evaluator
```

## RAGAS 指标说明

| 指标 | 含义 | 优化方向 |
|------|------|----------|
| **Context Precision** | 检索的记忆有多"精"（相关记忆占检索结果的比例） | 调整向量检索参数、改进 embedding 模型 |
| **Context Recall** | 检索的记忆有多"全"（相关记忆被召回的比例） | 增加检索数量 n_results、改进 embedding 模型 |
| **Faithfulness** | AI 是否"忠实"于检索到的记忆回答 | 改进提示词，强调约束条件 |
| **Answer Correctness** | AI 回答的准确性（对比标准答案） | 优化提示词或更换 LLM 模型 |

## 评测流程

1. **准备测试数据**: 保存测试用的长期记忆到向量库
2. **发送问题**: 通过 Go 接口发送测试问题
3. **获取结果**: 获取检索到的记忆 + AI 生成的回答
4. **计算指标**: 使用 RAGAS 计算各项指标
5. **分析优化**: 根据指标定位问题环节

## 接口详情

### Go 后端接口

**POST** `/api/v1/ai/evaluate-rag`

请求:
```json
{
  "user_id": "test_user_ragas_001",
  "question": "我之前有什么口吃症状？"
}
```

响应:
```json
{
  "question": "我之前有什么口吃症状？",
  "contexts": ["用户自述：我说话时第一个字总是重复..."],
  "context_ids": ["memory_1"],
  "answer": "根据您之前的记录，您有首字重复的症状...",
  "model_used": "doubao",
  "retrieval_time_ms": 45.2,
  "generate_time_ms": 1234.5
}
```

## 扩展：添加更多测试用例

编辑 `ragas_memory_evaluator.py` 中的 `run_evaluation` 方法:

```python
test_cases = [
    {
        "question": "你的新问题？",
        "expected_memories": ["应该被检索到的记忆关键词"],
        "expected_answer": "期望的回答"
    },
    # ... 更多测试用例
]
```

## 故障排查

| 问题 | 解决方式 |
|------|----------|
| RAGAS 未安装 | `pip install ragas` |
| Go API 连接失败 | 检查 Go 服务是否启动、端口是否正确 |
| 认证失败 | 检查 `GO_API_TOKEN` 是否有效 |
| 检索不到记忆 | 确认 Python 记忆服务已启动并保存了测试数据 |
