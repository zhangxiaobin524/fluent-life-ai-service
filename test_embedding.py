"""
测试脚本：演示文本嵌入功能
运行前确保：
1. 安装依赖: pip install -r requirements.txt
2. 设置环境变量: export DEEPSEEK_API_KEY="your-key"
"""

import os
import asyncio
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com"
)

# 测试用的口吃矫正资料
SAMPLE_TEXTS = [
    "口吃是一种言语流畅性障碍，表现为说话时出现重复、延长或阻塞。",
    "呼吸训练是口吃矫正的基础，要学会腹式呼吸，保持气息平稳。",
    "慢速说话可以有效减少口吃频率，建议每分钟60-80个字。",
    "轻柔起音技巧：说话时声带要放松，避免用力过猛导致卡顿。",
    "心理调节很重要，口吃者往往伴随焦虑和紧张情绪。",
]


def test_single_embedding():
    """测试单个文本嵌入"""
    print("=" * 50)
    print("📝 测试 1: 单个文本嵌入")
    print("=" * 50)
    
    text = "口吃矫正需要耐心和持续的练习。"
    
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        
        embedding = response.data[0].embedding
        
        print(f"输入文本: {text}")
        print(f"向量维度: {len(embedding)}")
        print(f"向量前10个值: {embedding[:10]}")
        print(f"向量后10个值: {embedding[-10:]}")
        print("✅ 单文本嵌入成功！\n")
        
        return embedding
        
    except Exception as e:
        print(f"❌ 失败: {e}\n")
        return None


def test_batch_embedding():
    """测试批量文本嵌入"""
    print("=" * 50)
    print("📝 测试 2: 批量文本嵌入")
    print("=" * 50)
    
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=SAMPLE_TEXTS
        )
        
        embeddings = [item.embedding for item in response.data]
        
        print(f"输入文本数: {len(SAMPLE_TEXTS)}")
        print(f"输出向量数: {len(embeddings)}")
        print(f"每个向量维度: {len(embeddings[0])}")
        
        for i, text in enumerate(SAMPLE_TEXTS):
            print(f"\n  [{i+1}] {text[:30]}...")
            print(f"      向量前5个值: {embeddings[i][:5]}")
        
        print("\n✅ 批量嵌入成功！")
        return embeddings
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        return None


def calculate_similarity(emb1, emb2):
    """计算两个向量的余弦相似度"""
    import numpy as np
    
    vec1 = np.array(emb1)
    vec2 = np.array(emb2)
    
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity


def test_similarity():
    """测试文本相似度计算"""
    print("\n" + "=" * 50)
    print("📝 测试 3: 文本相似度计算")
    print("=" * 50)
    
    texts = [
        "口吃矫正需要每天练习呼吸训练",
        "呼吸训练是口吃矫正的重要方法",
        "今天的天气真不错",
    ]
    
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        
        embeddings = [item.embedding for item in response.data]
        
        # 计算相似度
        sim_0_1 = calculate_similarity(embeddings[0], embeddings[1])
        sim_0_2 = calculate_similarity(embeddings[0], embeddings[2])
        
        print(f"文本1: {texts[0]}")
        print(f"文本2: {texts[1]}")
        print(f"  → 相似度: {sim_0_1:.4f} (应该较高)")
        
        print(f"\n文本1: {texts[0]}")
        print(f"文本3: {texts[2]}")
        print(f"  → 相似度: {sim_0_2:.4f} (应该较低)")
        
        print("\n✅ 相似度计算成功！")
        
    except Exception as e:
        print(f"❌ 失败: {e}")


if __name__ == "__main__":
    print("\n🚀 Fluent Life AI - Embedding 测试\n")
    
    # 检查 API Key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠️  警告: 未设置 DEEPSEEK_API_KEY 环境变量")
        print("请先设置: export DEEPSEEK_API_KEY='your-api-key'\n")
        exit(1)
    
    # 运行测试
    test_single_embedding()
    test_batch_embedding()
    test_similarity()
    
    print("\n" + "=" * 50)
    print("✅ 所有测试完成！")
    print("=" * 50)
