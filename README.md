# Fluent Life AI Service

Python AI 服务，提供文本嵌入(Embedding)、知识库管理、AI 文本处理等功能。
Go 后端通过 HTTP API 调用此服务。

## 🎯 为什么用 Python 做 AI 服务？

| 功能 | Python 优势 | Go 劣势 |
|------|------------|---------|
| **文本嵌入 (Embedding)** | OpenAI/DeepSeek SDK 完善，一行代码搞定 | 需要手动封装 HTTP 请求 |
| **向量数据库** | ChromaDB、Pinecone 等原生支持 | 客户端库不完善 |
| **大模型调用** | 丰富的 SDK (OpenAI, LangChain) | 生态不如 Python |
| **ML/AI 处理** | NumPy, SciPy, 各种模型 | 生态薄弱 |
| **快速原型** | 开发快，调试方便 | 适合高并发，但开发慢 |

**架构设计**:
```
Go Backend → HTTP API → Python AI Service → OpenAI/ChromaDB
     ↑                                              ↓
     └────────────  AI 处理结果  ←──────────────────┘
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd fluent-life-ai-service

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
```

### 3. 运行测试

```bash
# 测试文本嵌入功能
python test_embedding.py
```

### 4. 初始化知识库

```bash
# 将口吃矫正资料向量化存入 ChromaDB
python setup_knowledge_base.py
```

### 5. 启动服务

```bash
python main.py
```

服务启动后访问: http://localhost:8002/docs (Swagger UI)

## 📡 API 端点

### 1. 文本嵌入

```bash
curl -X POST "http://localhost:8002/embedding" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "口吃矫正需要持续练习",
    "model": "text-embedding-3-small"
  }'
```

响应:
```json
{
  "embedding": [0.023, -0.045, ...],  // 1536维向量
  "model": "text-embedding-3-small",
  "dimensions": 1536,
  "text_preview": "口吃矫正需要持续练习"
}
```

### 2. 批量嵌入

```bash
curl -X POST "http://localhost:8002/embedding/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["文本1", "文本2", "文本3"]
  }'
```

### 3. 添加到知识库

```bash
curl -X POST "http://localhost:8002/knowledge-base/add" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "stutter_correction",
    "documents": ["新的口吃矫正资料..."],
    "metadatas": [{"category": "训练方法"}]
  }'
```

### 4. 搜索知识库

```bash
curl -X POST "http://localhost:8002/knowledge-base/search" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "stutter_correction",
    "query": "怎么练习呼吸",
    "n_results": 3
  }'
```

响应:
```json
{
  "query": "怎么练习呼吸",
  "results": [
    {
      "id": "breathing_001",
      "document": "腹式呼吸训练方法...",
      "metadata": {"category": "呼吸训练"},
      "distance": 0.234
    }
  ]
}
```

## 🔗 Go 后端调用示例

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
)

type EmbeddingRequest struct {
    Text  string `json:"text"`
    Model string `json:"model,omitempty"`
}

type EmbeddingResponse struct {
    Embedding    []float64 `json:"embedding"`
    Model        string    `json:"model"`
    Dimensions   int       `json:"dimensions"`
    TextPreview  string    `json:"text_preview"`
}

func GetEmbedding(text string) ([]float64, error) {
    reqBody := EmbeddingRequest{Text: text}
    jsonData, _ := json.Marshal(reqBody)
    
    resp, err := http.Post(
        "http://localhost:8002/embedding",
        "application/json",
        bytes.NewBuffer(jsonData),
    )
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    var result EmbeddingResponse
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, err
    }
    
    return result.Embedding, nil
}
```

## 📁 文件结构

```
fluent-life-ai-service/
├── main.py                  # FastAPI 主服务
├── test_embedding.py        # Embedding 测试脚本
├── setup_knowledge_base.py  # 知识库初始化脚本
├── requirements.txt         # Python 依赖
├── .env.example            # 环境变量示例
└── chroma_db/              # 向量数据库（自动创建）
```

## 🛠️ 开发计划

- [x] 文本嵌入 (Embedding) API
- [x] 批量嵌入
- [x] ChromaDB 知识库
- [x] 知识库搜索
- [ ] 口吃矫正 AI 对话
- [ ] 语音转文本 (ASR)
- [ ] 文本转语音 (TTS)
- [ ] 实时语音分析

## 📚 参考资料

- [DeepSeek API 文档](https://platform.deepseek.com/)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
# fluent-life-ai-service
