"""
Fluent Life AI Service
提供文本嵌入(Embedding)、知识库管理、AI 文本处理等服务
使用 DeepSeek API 生成 Embedding
"""

import os
import asyncio
import time
import uuid
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import numpy as np

# OpenTelemetry 链路追踪
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract, set_global_textmap
from opentelemetry.propagators.b3 import B3Format
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# 加载环境变量
load_dotenv()

# ==================== OpenTelemetry 初始化 ====================
def init_tracer():
    """初始化 OpenTelemetry Tracer"""
    jaeger_endpoint = os.getenv("JAEGER_ENDPOINT", "localhost:4318")
    
    # 创建 OTLP exporter
    exporter = OTLPSpanExporter(
        endpoint=f"http://{jaeger_endpoint}/v1/traces",
    )
    
    # 创建 provider（设置 service name）
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    resource = Resource.create({SERVICE_NAME: "fluent-life-ai"})
    provider = TracerProvider(resource=resource)
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    
    # 设置全局 provider
    trace.set_tracer_provider(provider)
    
    # 设置 propagator（支持 W3C Trace Context 和 B3）
    set_global_textmap(TraceContextTextMapPropagator())
    
    print(f"✅ OpenTelemetry 初始化完成，上报地址: {jaeger_endpoint}")
    return trace.get_tracer("fluent-life-ai")

# 初始化 tracer
tracer = init_tracer()

# ==================== 配置 ====================
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL_NAME = "text-embedding-3-small"  # DeepSeek 使用的 OpenAI embedding 模型
EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-small 的维度

# DeepSeek/OpenAI 客户端
embedding_client = None

# ChromaDB 客户端
chroma_client = None


# ==================== 生命周期管理 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global chroma_client, embedding_client
    
    # 启动时初始化 DeepSeek 客户端
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️ 警告: 未设置 DEEPSEEK_API_KEY，Embedding 功能将不可用")
        embedding_client = None
    else:
        embedding_client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        print(f"✅ DeepSeek 客户端初始化完成，Embedding 模型: {EMBEDDING_MODEL_NAME}")
    
    # 启动时初始化 ChromaDB
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    print(f"✅ ChromaDB 初始化完成，路径: {CHROMA_DB_PATH}")
    
    yield
    
    # 关闭时清理
    print("🛑 服务关闭")


# ==================== FastAPI 应用 ====================
app = FastAPI(
    title="Fluent Life AI Service",
    description="口吃矫正 AI 服务 - 提供文本嵌入、知识库管理（DeepSeek API）",
    version="1.0.0",
    lifespan=lifespan
)

# ==================== 链路追踪中间件 ====================
@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    """
    从请求头中提取 trace context，创建 span
    支持从 Go 服务传递过来的 traceparent header
    """
    # 从 header 提取 trace context
    context = extract(request.headers)
    
    # 创建 span
    with tracer.start_as_current_span(
        f"{request.method} {request.url.path}",
        context=context,
        kind=trace.SpanKind.SERVER,
    ) as span:
        # 记录请求信息
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.url", str(request.url))
        span.set_attribute("http.route", request.url.path)
        span.set_attribute("http.host", request.headers.get("host", ""))
        span.set_attribute("http.user_agent", request.headers.get("user-agent", ""))
        span.set_attribute("service.name", "fluent-life-ai")
        
        # 执行请求
        start_time = time.time()
        try:
            response = await call_next(request)
            # 记录响应状态
            span.set_attribute("http.status_code", response.status_code)
            span.set_attribute("http.duration_ms", int((time.time() - start_time) * 1000))
            return response
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            raise


# ==================== 数据模型 ====================
class EmbeddingRequest(BaseModel):
    """文本嵌入请求"""
    text: str


class EmbeddingResponse(BaseModel):
    """文本嵌入响应"""
    embedding: List[float]
    model: str
    dimensions: int
    text_preview: str


class BatchEmbeddingRequest(BaseModel):
    """批量文本嵌入请求"""
    texts: List[str]


class BatchEmbeddingResponse(BaseModel):
    """批量文本嵌入响应"""
    embeddings: List[List[float]]
    model: str
    count: int


class AddToKnowledgeBaseRequest(BaseModel):
    """添加到知识库请求"""
    collection_name: str = "stutter_correction"
    documents: List[str]
    ids: Optional[List[str]] = None
    metadatas: Optional[List[dict]] = None


class SearchKnowledgeBaseRequest(BaseModel):
    """知识库搜索请求"""
    collection_name: str = "stutter_correction"
    query: str
    n_results: int = 5


# ==================== API 端点 ====================
@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "fluent-life-ai",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "dimensions": EMBEDDING_DIMENSIONS,
        "embedding_client_ready": embedding_client is not None
    }


def get_embedding(text: str) -> List[float]:
    """调用 DeepSeek API 生成 embedding"""
    if not embedding_client:
        raise HTTPException(status_code=503, detail="Embedding 服务未初始化，请检查 DEEPSEEK_API_KEY")
    
    response = embedding_client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=text
    )
    return response.data[0].embedding


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """批量调用 DeepSeek API 生成 embeddings"""
    if not embedding_client:
        raise HTTPException(status_code=503, detail="Embedding 服务未初始化，请检查 DEEPSEEK_API_KEY")
    
    response = embedding_client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=texts
    )
    return [item.embedding for item in response.data]


@app.post("/embedding", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """将文本转换为向量嵌入"""
    try:
        embedding = get_embedding(request.text)
        return EmbeddingResponse(
            embedding=embedding,
            model=EMBEDDING_MODEL_NAME,
            dimensions=len(embedding),
            text_preview=request.text[:50] + "..." if len(request.text) > 50 else request.text
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding 失败: {str(e)}")


@app.post("/embedding/batch", response_model=BatchEmbeddingResponse)
async def create_batch_embeddings(request: BatchEmbeddingRequest):
    """批量文本嵌入"""
    try:
        embeddings = get_embeddings(request.texts)
        return BatchEmbeddingResponse(
            embeddings=embeddings,
            model=EMBEDDING_MODEL_NAME,
            count=len(request.texts)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量 Embedding 失败: {str(e)}")


@app.post("/knowledge-base/add")
async def add_to_knowledge_base(request: AddToKnowledgeBaseRequest):
    """添加文档到知识库"""
    try:
        collection = chroma_client.get_or_create_collection(
            name=request.collection_name,
            metadata={"description": "口吃矫正知识库"}
        )
        
        embeddings = get_embeddings(request.documents)
        ids = request.ids or [f"doc_{i}" for i in range(len(request.documents))]
        
        collection.add(
            embeddings=embeddings,
            documents=request.documents,
            ids=ids,
            metadatas=request.metadatas
        )
        
        return {
            "success": True,
            "collection": request.collection_name,
            "added_count": len(request.documents),
            "ids": ids
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加到知识库失败: {str(e)}")


def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


@app.post("/knowledge-base/search")
async def search_knowledge_base(request: SearchKnowledgeBaseRequest):
    """语义搜索知识库"""
    try:
        collection = chroma_client.get_collection(name=request.collection_name)
        query_embedding = get_embedding(request.query)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # 获取结果
        ids = results['ids'][0]
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else [None] * len(ids)
        distances = results['distances'][0]
        
        # 格式化结果
        THRESHOLD = 0.8
        formatted_results = []
        for i in range(len(ids)):
            distance = distances[i]
            if distance <= THRESHOLD:
                formatted_results.append({
                    "id": ids[i],
                    "document": documents[i],
                    "metadata": metadatas[i],
                    "distance": distance,
                    "relevance": "high" if distance < 0.2 else "medium"
                })
        
        return {
            "query": request.query,
            "total_found": len(formatted_results),
            "relevant_results": len(formatted_results),
            "results": formatted_results,
            "collection": request.collection_name
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


# ==================== RAG 对话接口（带完整上下文追踪）====================

class RAGChatRequest(BaseModel):
    """RAG 对话请求"""
    question: str
    collection_name: str = "stutter_correction"
    n_results: int = 3
    user_context: Optional[str] = None


class RAGChatResponse(BaseModel):
    """RAG 对话响应（包含评测所需全部字段）"""
    question: str
    contexts: List[str]
    context_ids: List[str]
    answer: str
    model_used: str
    retrieval_time_ms: float
    generation_time_ms: float


async def retrieve_contexts(question: str, collection_name: str, n_results: int = 3) -> tuple:
    """
    检索相关上下文
    返回: (contexts列表, context_ids列表, retrieval_time_ms)
    """
    start_time = time.time()
    
    try:
        collection = chroma_client.get_collection(name=collection_name)
        query_embedding = get_embedding(question)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # 提取检索结果
        contexts = []
        context_ids = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                if doc:
                    contexts.append(doc)
                    context_ids.append(results['ids'][0][i] if results['ids'] else f"doc_{i}")
        
        retrieval_time = (time.time() - start_time) * 1000
        return contexts, context_ids, retrieval_time
        
    except Exception as e:
        print(f"检索失败: {e}")
        return [], [], 0


async def generate_answer(question: str, contexts: List[str], user_context: Optional[str] = None) -> tuple:
    """
    基于上下文生成回答
    返回: (answer, model_used, generation_time_ms)
    """
    start_time = time.time()
    
    # 构建系统提示
    system_prompt = """你是 Fluent Life 口吃矫正助手，基于以下参考资料回答用户问题。
请确保回答：
1. 基于提供的参考资料
2. 专业、准确、易懂
3. 如果不确定，请说明"根据现有资料无法确定"

参考资料：
"""
    
    # 添加上下文
    for i, ctx in enumerate(contexts, 1):
        system_prompt += f"\n[{i}] {ctx}\n"
    
    # 添加用户背景（如果有）
    if user_context:
        system_prompt += f"\n用户背景信息：{user_context}\n"
    
    try:
        # 使用项目中已配置的 embedding_client (DeepSeek)
        if embedding_client is None:
            return "错误：LLM 客户端未初始化", "error", 0
        
        response = embedding_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        answer = response.choices[0].message.content
        generation_time = (time.time() - start_time) * 1000
        
        return answer, "deepseek-chat", generation_time
        
    except Exception as e:
        print(f"生成失败: {e}")
        return "抱歉，生成回答时出现错误。", "error", (time.time() - start_time) * 1000


@app.post("/rag/chat", response_model=RAGChatResponse)
async def rag_chat(request: RAGChatRequest):
    """
    RAG 对话接口 - 完整记录检索和生成过程
    用于：日常对话 + RAGAS 评测数据采集
    """
    try:
        # 1. 检索相关上下文
        contexts, context_ids, retrieval_time = await retrieve_contexts(
            question=request.question,
            collection_name=request.collection_name,
            n_results=request.n_results
        )
        
        # 2. 生成回答
        answer, model_used, generation_time = await generate_answer(
            question=request.question,
            contexts=contexts,
            user_context=request.user_context
        )
        
        return RAGChatResponse(
            question=request.question,
            contexts=contexts,
            context_ids=context_ids,
            answer=answer,
            model_used=model_used,
            retrieval_time_ms=round(retrieval_time, 2),
            generation_time_ms=round(generation_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG对话失败: {str(e)}")


@app.get("/knowledge-base/collections")
async def list_collections():
    """列出所有知识库集合"""
    try:
        collections = chroma_client.list_collections()
        return {
            "collections": [c.name for c in collections]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取集合列表失败: {str(e)}")


# ==================== 长期记忆系统 ====================

class SaveMemoryRequest(BaseModel):
    """保存长期记忆请求"""
    user_id: str
    dialogue: str
    topic: str = "一般对话"
    timestamp: str


class SearchMemoryRequest(BaseModel):
    """搜索长期记忆请求"""
    user_id: str
    query: str
    n_results: int = 3


# 长期记忆集合名称
LONG_TERM_MEMORY_COLLECTION = "user_long_term_memory"


def get_or_create_memory_collection():
    """获取或创建长期记忆集合"""
    try:
        collection = chroma_client.get_collection(name=LONG_TERM_MEMORY_COLLECTION)
    except Exception:
        collection = chroma_client.create_collection(
            name=LONG_TERM_MEMORY_COLLECTION,
            metadata={"description": "用户长期对话记忆"}
        )
    return collection


@app.post("/memory/save")
async def save_long_term_memory(request: SaveMemoryRequest):
    """
    保存对话到长期记忆（向量库）
    
    流程：
    1. 将对话内容转为向量
    2. 存入 ChromaDB，附带用户ID、话题、时间等元数据
    3. 后续可通过向量相似度搜索相关历史
    """
    try:
        collection = get_or_create_memory_collection()
        
        # 生成唯一ID
        memory_id = f"{request.user_id}_{int(time.time() * 1000)}"
        
        # 将对话转为向量
        dialogue_embedding = get_embedding(request.dialogue)
        
        # 存入向量库
        collection.add(
            embeddings=[dialogue_embedding],
            documents=[request.dialogue],
            metadatas=[{
                "user_id": request.user_id,
                "topic": request.topic,
                "timestamp": request.timestamp,
                "type": "dialogue"
            }],
            ids=[memory_id]
        )
        
        print(f"💾 长期记忆已保存: user={request.user_id}, topic={request.topic}")
        
        return {
            "success": True,
            "memory_id": memory_id,
            "user_id": request.user_id,
            "topic": request.topic
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存长期记忆失败: {str(e)}")


@app.post("/memory/search")
async def search_long_term_memory(request: SearchMemoryRequest):
    """
    搜索长期记忆（向量相似度检索）
    
    流程：
    1. 将查询转为向量
    2. 在向量库中搜索相似对话（只搜索该用户的）
    3. 返回最相关的N条历史对话
    """
    try:
        collection = get_or_create_memory_collection()
        
        # 将查询转为向量
        query_embedding = get_embedding(request.query)
        
        # 向量搜索，只搜索该用户的记忆
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.n_results,
            where={"user_id": request.user_id},  # 过滤条件：只查该用户
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化结果
        memories = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                memories.append({
                    "dialogue": results['documents'][0][i],
                    "topic": results['metadatas'][0][i].get("topic", "未知"),
                    "timestamp": results['metadatas'][0][i].get("timestamp", ""),
                    "distance": results['distances'][0][i]
                })
        
        print(f"🔍 长期记忆搜索: user={request.user_id}, query={request.query[:20]}..., found={len(memories)}")
        
        return {
            "success": True,
            "user_id": request.user_id,
            "query": request.query,
            "memories": memories,
            "count": len(memories)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索长期记忆失败: {str(e)}")


@app.get("/memory/stats/{user_id}")
async def get_memory_stats(user_id: str):
    """获取用户长期记忆统计"""
    try:
        collection = get_or_create_memory_collection()
        
        # 查询该用户的所有记忆
        results = collection.get(
            where={"user_id": user_id},
            include=["metadatas"]
        )
        
        # 统计话题分布
        topic_count = {}
        for metadata in results['metadatas']:
            topic = metadata.get("topic", "未知")
            topic_count[topic] = topic_count.get(topic, 0) + 1
        
        return {
            "user_id": user_id,
            "total_memories": len(results['ids']),
            "topic_distribution": topic_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取记忆统计失败: {str(e)}")


# ==================== 通用AI生成接口 ====================
class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = "You are a helpful assistant."
    model: str = "doubao-pro-32k-241215"
    max_tokens: int = 2000
    temperature: float = 0.3


@app.post("/api/v1/ai/generate")
async def generate_text(request: GenerateRequest):
    """
    通用文本生成接口 - 调用豆包API
    Go后端负责业务逻辑，此接口只负责纯AI调用
    """
    try:
        from volcenginesdkarkruntime import Ark
        
        client = Ark(base_url="https://ark.cn-beijing.volces.com/api/v3")
        
        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.prompt}
            ],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        
        content = response.choices[0].message.content
        
        return {
            "success": True,
            "content": content
        }
        
    except Exception as e:
        print(f"❌ AI生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI生成失败: {str(e)}")


# ==================== 训练计划工作流接口 ====================
from workflows.training_plan_workflow import get_workflow, TrainingPlanState

class StartPlanWorkflowRequest(BaseModel):
    user_id: str
    user_profile: dict
    training_history: Optional[List[dict]] = []


class WorkflowFeedbackRequest(BaseModel):
    thread_id: str
    feedback: str  # "too_easy", "too_hard", "adjust_focus", "good"
    feedback_detail: Optional[str] = ""


@app.post("/workflow/training-plan/start")
async def start_training_plan_workflow(request: StartPlanWorkflowRequest):
    """
    启动训练计划生成工作流
    返回初步计划和 thread_id（用于后续反馈）
    """
    try:
        workflow = get_workflow()
        
        # 初始化状态
        thread_id = str(uuid.uuid4())
        initial_state: TrainingPlanState = {
            "thread_id": thread_id,
            "user_id": request.user_id,
            "user_profile": request.user_profile,
            "training_history": request.training_history or [],
            "current_plan": None,
            "plan_content": None,
            "feedback": None,
            "feedback_detail": None,
            "adjustment_count": 0,
            "max_adjustments": 3,
            "status": "running",
            "conversation_history": [],
            "final_plan": None,
            "final_plan_id": None
        }
        
        # 执行工作流
        result = workflow.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        return {
            "success": True,
            "thread_id": thread_id,
            "status": result["status"],
            "plan": result["current_plan"],
            "plan_content": result["plan_content"],
            "conversation_history": result["conversation_history"],
            "message": "请查看计划并提供反馈"
        }
        
    except Exception as e:
        print(f"❌ 启动工作流失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动工作流失败: {str(e)}")


@app.post("/workflow/training-plan/feedback")
async def submit_workflow_feedback(request: WorkflowFeedbackRequest):
    """
    提交反馈，继续工作流
    """
    try:
        workflow = get_workflow()
        
        # 恢复状态并更新反馈
        result = workflow.invoke(
            {
                "feedback": request.feedback,
                "feedback_detail": request.feedback_detail or ""
            },
            config={"configurable": {"thread_id": request.thread_id}}
        )
        
        response = {
            "success": True,
            "thread_id": request.thread_id,
            "status": result["status"],
            "plan": result["current_plan"],
            "plan_content": result["plan_content"],
            "conversation_history": result["conversation_history"],
            "adjustment_count": result["adjustment_count"]
        }
        
        # 如果已完成，添加最终计划
        if result["status"] == "completed":
            response["final_plan"] = result["final_plan"]
            response["message"] = "计划已确认并保存"
        else:
            response["message"] = "请继续提供反馈"
        
        return response
        
    except Exception as e:
        print(f"❌ 提交反馈失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")


@app.get("/workflow/training-plan/{thread_id}")
async def get_workflow_status(thread_id: str):
    """
    获取工作流状态
    """
    try:
        workflow = get_workflow()
        
        # 获取状态
        state = workflow.get_state({"configurable": {"thread_id": thread_id}})
        
        return {
            "success": True,
            "thread_id": thread_id,
            "status": state.values.get("status", "unknown"),
            "plan": state.values.get("current_plan"),
            "adjustment_count": state.values.get("adjustment_count", 0),
            "conversation_history": state.values.get("conversation_history", [])
        }
        
    except Exception as e:
        print(f"❌ 获取状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


# ==================== 专家团队工作流接口 ====================
from workflows.expert_team_workflow import get_expert_workflow, ExpertTeamState

class ExpertTeamRequest(BaseModel):
    """专家团队请求"""
    user_id: str
    user_message: str
    user_profile: Optional[dict] = {}


class ExpertTeamResponse(BaseModel):
    """专家团队响应"""
    success: bool
    thread_id: str
    question_type: str
    complexity: str
    experts_involved: List[str]
    execution_path: List[str]
    final_response: str


@app.post("/workflow/expert-team/chat", response_model=ExpertTeamResponse)
async def expert_team_chat(request: ExpertTeamRequest):
    """
    专家团队模式 - 智能路由到不同专家
    
    流程：
    1. 路由专家分析问题类型和复杂度
    2. 根据条件判断调用不同专家组合
    3. 汇总专家意见生成最终回复
    
    条件路由逻辑：
    - simple问题 -> 直接汇总（快速回复）
    - psychological -> 深度心理支持
    - data_analysis -> 数据分析师
    - training_plan -> 方案专家
    - diagnosis -> 诊断专家 -> 方案专家
    - complex问题 -> 增加数据分析和心理支持
    """
    try:
        workflow = get_expert_workflow()
        
        # 初始化状态
        thread_id = str(uuid.uuid4())
        initial_state: ExpertTeamState = {
            "thread_id": thread_id,
            "user_id": request.user_id,
            "user_message": request.user_message,
            "user_profile": request.user_profile or {},
            "question_type": "general",
            "complexity": "simple",
            "routing_reason": "",
            "diagnosis_result": None,
            "data_analysis_result": None,
            "plan_result": None,
            "support_result": None,
            "experts_involved": [],
            "execution_path": [],
            "final_response": "",
            "status": "running"
        }
        
        # 执行工作流
        result = workflow.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        return {
            "success": True,
            "thread_id": thread_id,
            "question_type": result.get("question_type", "general"),
            "complexity": result.get("complexity", "simple"),
            "experts_involved": result.get("experts_involved", []),
            "execution_path": result.get("execution_path", []),
            "final_response": result.get("final_response", "")
        }
        
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"❌ 专家团队工作流失败: {error_detail}")
        raise HTTPException(status_code=500, detail=f"专家团队处理失败: {str(e)}")


@app.get("/workflow/expert-team/{thread_id}")
async def get_expert_team_status(thread_id: str):
    """
    获取专家团队工作流状态
    """
    try:
        workflow = get_expert_workflow()
        
        # 获取状态
        state = workflow.get_state({"configurable": {"thread_id": thread_id}})
        
        return {
            "success": True,
            "thread_id": thread_id,
            "status": state.values.get("status", "unknown"),
            "question_type": state.values.get("question_type"),
            "complexity": state.values.get("complexity"),
            "experts_involved": state.values.get("experts_involved", []),
            "execution_path": state.values.get("execution_path", []),
            "final_response": state.values.get("final_response")
        }
        
    except Exception as e:
        print(f"❌ 获取专家团队状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


# ==================== Plan & Execute 工作流接口 ====================
from workflows.plan_execute_workflow import (
    get_plan_execute_workflow, 
    task_manager, 
    run_plan_execute_with_updates
)
import threading

class PlanExecuteRequest(BaseModel):
    """Plan & Execute 请求"""
    user_id: str
    user_message: str
    user_profile: Optional[Dict[str, Any]] = None


@app.post("/workflow/plan-execute/create")
async def plan_execute_create(request: PlanExecuteRequest):
    """
    创建 Plan & Execute 异步任务
    返回 task_id，用于后续轮询查询进度
    
    示例场景：
    - "帮我制定三个月口吃矫正方案"
    - "我想系统性地改善，给个完整计划"
    """
    try:
        thread_id = str(uuid.uuid4())
        
        # 创建任务记录
        task_id = task_manager.create_task(
            thread_id=thread_id,
            user_id=request.user_id,
            user_message=request.user_message
        )
        
        # 初始化状态
        initial_state = {
            "thread_id": thread_id,
            "user_id": request.user_id,
            "user_message": request.user_message,
            "user_profile": request.user_profile or {},
            "plan": [],
            "results": [],
            "current_step": 0,
            "total_steps": 0,
            "stream_callback": None,
            "final_response": "",
            "execution_path": [],
            "status": "running"
        }
        
        # 在后台线程中执行工作流
        def run_in_background():
            try:
                run_plan_execute_with_updates(task_id, initial_state)
            except Exception as e:
                print(f"❌ 后台任务执行失败: {str(e)}")
        
        thread = threading.Thread(target=run_in_background)
        thread.daemon = True
        thread.start()
        
        return {
            "success": True,
            "task_id": task_id,
            "status": "planning",
            "message": "任务已创建，开始规划中..."
        }
        
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"❌ 创建 Plan & Execute 任务失败: {error_detail}")
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")


@app.get("/workflow/plan-execute/status/{task_id}")
async def plan_execute_status(task_id: str):
    """
    查询 Plan & Execute 任务状态和进度
    前端轮询此接口获取实时进度
    """
    try:
        task = task_manager.get_task(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return {
            "success": True,
            "task_id": task_id,
            "status": task.get("status"),  # planning, executing, aggregating, completed, failed
            "plan": task.get("plan", []),
            "current_step": task.get("current_step", 0),
            "total_steps": task.get("total_steps", 0),
            "step_results": task.get("step_results", []),
            "execution_path": task.get("execution_path", []),
            "final_response": task.get("final_response"),
            "error": task.get("error"),
            "updated_at": task.get("updated_at").isoformat() if task.get("updated_at") else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 查询任务状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询状态失败: {str(e)}")


# 保留原来的同步接口，供兼容使用
@app.post("/workflow/plan-execute/chat")
async def plan_execute_chat(request: PlanExecuteRequest):
    """
    Plan & Execute 模式 - 同步版本（兼容旧接口）
    """
    try:
        workflow = get_plan_execute_workflow()
        
        thread_id = str(uuid.uuid4())
        
        initial_state = {
            "thread_id": thread_id,
            "user_id": request.user_id,
            "user_message": request.user_message,
            "user_profile": request.user_profile or {},
            "plan": [],
            "results": [],
            "current_step": 0,
            "total_steps": 0,
            "stream_callback": None,
            "final_response": "",
            "execution_path": [],
            "status": "running"
        }
        
        result = workflow.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        return {
            "success": True,
            "thread_id": thread_id,
            "plan": result.get("plan", []),
            "step_results": result.get("results", []),
            "execution_path": result.get("execution_path", []),
            "final_response": result.get("final_response", "")
        }
        
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"❌ Plan & Execute 工作流失败: {error_detail}")
        raise HTTPException(status_code=500, detail=f"计划执行失败: {str(e)}")


# ==================== 面试双Agent系统接口 ====================
# Day 21: LangGraph 异步多 Agent 面试工作流
from workflows.interview_langgraph import async_interview_manager as interview_manager

class InterviewStartRequest(BaseModel):
    """开始面试请求"""
    candidate_name: str
    position: str = "软件工程师"


class InterviewAnswerRequest(BaseModel):
    """提交回答请求"""
    session_id: str
    answer: str


@app.post("/interview/start")
async def interview_start(request: InterviewStartRequest):
    """
    开始双Agent面试
    - 初始化 InterviewerAgent 和 ObserverAgent
    - 返回第一个问题和 session_id
    """
    try:
        session_id = str(uuid.uuid4())
        
        result = await interview_manager.start_interview(
            session_id=session_id,
            candidate_name=request.candidate_name,
            position=request.position
        )
        
        return {
            "code": 0,
            "message": "面试开始",
            "data": result
        }
        
    except Exception as e:
        import traceback
        print(f"❌ 开始面试失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"开始面试失败: {str(e)}")


@app.post("/interview/answer")
async def interview_answer(request: InterviewAnswerRequest):
    """
    提交候选人回答
    - ObserverAgent 分析回答质量
    - InterviewerAgent 生成下一个问题
    - 返回新问题和当前观察结果
    """
    try:
        result = await interview_manager.submit_answer(
            session_id=request.session_id,
            answer=request.answer
        )
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return {
            "code": 0,
            "message": "处理成功",
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ 处理回答失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"处理回答失败: {str(e)}")


@app.get("/interview/report/{session_id}")
async def interview_report(session_id: str):
    """
    获取观察员完整报告
    面试结束后调用，获取 ObserverAgent 的详细分析
    """
    try:
        state = interview_manager.get_session_state(session_id)
        
        if not state:
            raise HTTPException(status_code=404, detail="面试会话不存在")
        
        result = {
            "session_id": session_id,
            "candidate_name": state.get("candidate_name", ""),
            "position": state.get("position", ""),
            "total_questions": state.get("question_index", 0),
            "observations": state.get("observations_history", []),
            "final_report": state.get("final_report")
        }
        
        return {
            "code": 0,
            "message": "获取成功",
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 获取报告失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取报告失败: {str(e)}")


@app.get("/interview/observation/{session_id}")
async def interview_observation(session_id: str):
    """
    获取当前观察结果
    用于后台异步获取 ObserverAgent 的分析结果
    """
    try:
        state = interview_manager.get_session_state(session_id)
        
        if not state:
            raise HTTPException(status_code=404, detail="面试会话不存在")
        
        # 获取待处理的观察结果
        pending_observation = state.get("pending_observation")
        
        return {
            "code": 0,
            "message": "获取成功",
            "data": {
                "session_id": session_id,
                "current_observation": pending_observation,
                "has_observation": pending_observation is not None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 获取观察结果失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取观察结果失败: {str(e)}")


class InterviewEndRequest(BaseModel):
    """结束面试请求"""
    session_id: str


@app.post("/interview/end")
async def interview_end(request: InterviewEndRequest):
    """
    结束面试
    返回完整的 ObserverAgent 观察报告
    """
    try:
        result = await interview_manager.end_interview(request.session_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return {
            "code": 0,
            "message": "面试结束",
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 结束面试失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"结束面试失败: {str(e)}")


# ==================== 启动 ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
