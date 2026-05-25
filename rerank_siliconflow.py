"""
SiliconFlow Re-Rank API 封装
按量付费，无需本地模型
文档：https://docs.siliconflow.cn/api-reference/rerank/create-rerank
"""
import os
import httpx
from typing import List, Dict

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
RERANK_API_URL = "https://api.siliconflow.cn/v1/rerank"


async def rerank_with_api(
    query: str,
    documents: List[str],
    model: str = "BAAI/bge-reranker-v2-m3"
) -> List[Dict]:
    """
    使用 SiliconFlow API 进行重排序
    
    Args:
        query: 用户查询
        documents: 文档列表
        model: 重排序模型
        
    Returns:
        按相关性排序的文档列表
    """
    if not SILICONFLOW_API_KEY:
        raise ValueError("缺少 SILICONFLOW_API_KEY 环境变量")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            RERANK_API_URL,
            headers={
                "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
                "return_documents": True
            }
        )
        response.raise_for_status()
        data = response.json()
        
        # 解析结果
        results = []
        for item in data["results"]:
            results.append({
                "index": item["index"],
                "document": item["document"],
                "rerank_score": item["relevance_score"],
                "relevance": "high" if item["relevance_score"] > 0.8 else "medium" if item["relevance_score"] > 0.5 else "low"
            })
        
        return results


# 价格参考（2024年）:
# BAAI/bge-reranker-v2-m3: ￥0.5 / 1M tokens
