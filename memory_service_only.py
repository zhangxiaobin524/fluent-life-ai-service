"""
简化版向量库服务 - 仅用于 RAGAS 评估
提供 /memory/save 和 /memory/search 接口
"""

import os
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI

load_dotenv()

app = FastAPI(title="Memory Service - Simplified")

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

embedding_client = None
chroma_client = None

# ==================== 请求模型 ====================
class LongTermMemoryRequest(BaseModel):
    user_id: str
    dialogue: str
    topic: str = "general"
    timestamp: Optional[str] = None

class SearchLongTermMemoryRequest(BaseModel):
    user_id: str
    query: str
    n_results: int = 5

# ==================== Embedding ====================
def get_embedding(text: str) -> List[float]:
    """获取文本的 embedding 向量"""
    global embedding_client
    
    if embedding_client is None:
        # 如果没有配置 API Key，使用 Mock
        return [0.1] * EMBEDDING_DIMENSIONS
    
    try:
        response = embedding_client.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=text[:8000]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding 生成失败: {e}, 使用 Mock")
        return [0.1] * EMBEDDING_DIMENSIONS

# ==================== API 端点 ====================
@app.post("/memory/save")
async def save_long_term_memory(req: LongTermMemoryRequest):
    """保存对话到长期记忆"""
    global chroma_client
    
    try:
        # 获取或创建集合
        collection_name = f"memories_{req.user_id}"
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # 生成 embedding
        embedding = get_embedding(req.dialogue)
        
        # 生成唯一 ID
        import uuid
        memory_id = f"mem_{uuid.uuid4().hex[:8]}"
        
        # 添加到向量库
        collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[req.dialogue],
            metadatas=[{
                "topic": req.topic,
                "timestamp": req.timestamp or "2024-01-01T00:00:00Z"
            }]
        )
        
        return {"status": "success", "memory_id": memory_id}
        
    except Exception as e:
        print(f"保存记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/search")
async def search_long_term_memory(req: SearchLongTermMemoryRequest):
    """搜索长期记忆"""
    global chroma_client
    
    try:
        collection_name = f"memories_{req.user_id}"
        
        # 检查集合是否存在
        try:
            collection = chroma_client.get_collection(collection_name)
        except Exception:
            # 兼容 Go 端的期望格式
            return {"results": [], "documents": [], "count": 0}
        
        # 生成查询 embedding
        query_embedding = get_embedding(req.query)
        
        # 搜索
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=req.n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化结果 - 兼容两种格式
        memories = []
        documents = []  # Go 端期望的格式
        for i in range(len(results['ids'][0])):
            doc = results['documents'][0][i]
            documents.append(doc)
            memories.append({
                "id": results['ids'][0][i],
                "dialogue": doc,
                "metadata": results['metadatas'][0][i],
                "similarity": 1 - results['distances'][0][i]  # 转换为相似度
            })
        
        return {"results": memories, "documents": documents, "count": len(memories)}
        
    except Exception as e:
        print(f"搜索记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "memory-only"}

# ==================== 启动时初始化 ====================
@app.on_event("startup")
async def startup():
    global chroma_client, embedding_client
    
    # 初始化 ChromaDB
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    print(f"✅ ChromaDB 初始化完成，路径: {CHROMA_DB_PATH}")
    
    # 初始化 Embedding 客户端
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        embedding_client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        print(f"✅ DeepSeek 客户端初始化完成")
    else:
        print("⚠️ 未配置 DEEPSEEK_API_KEY，使用 Mock Embedding")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
