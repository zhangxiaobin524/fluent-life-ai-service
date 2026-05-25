"""
口吃矫正知识库初始化脚本
将口吃矫正资料向量化存入 ChromaDB
"""

import os
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 初始化
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com"
)

chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

# 口吃矫正资料
STUTTER_CORRECTION_MATERIALS = [
    {
        "id": "breathing_001",
        "content": """腹式呼吸训练方法：
1. 平躺，一手放胸部，一手放腹部
2. 吸气时腹部隆起，胸部保持不动
3. 呼气时腹部收缩，像气球放气
4. 每天练习10分钟，养成腹式呼吸习惯
5. 说话时保持这种呼吸方式""",
        "metadata": {"category": "呼吸训练", "level": "基础", "type": "训练方法"}
    },
    {
        "id": "slow_speech_001",
        "content": """慢速说话技巧：
1. 将语速控制在每分钟60-80个字
2. 每个字之间留出0.2-0.3秒停顿
3. 不要急于说完，给自己充足时间
4. 可以使用节拍器辅助练习
5. 逐渐建立新的说话节奏""",
        "metadata": {"category": "语速控制", "level": "基础", "type": "技巧"}
    },
    {
        "id": "gentle_onset_001",
        "content": """轻柔起音训练：
1. 避免说话起始时声带过度紧张
2. 用"哈"音练习，感受声带放松状态
3. 元音开头时，先发出轻微的气流声
4. 想象声音是从喉咙深处缓缓流出
5. 每天练习5-10分钟，形成肌肉记忆""",
        "metadata": {"category": "发音技巧", "level": "中级", "type": "训练方法"}
    },
    {
        "id": "psychology_001",
        "content": """口吃心理调节：
1. 接受口吃是正常的言语现象，不要过度焦虑
2. 降低自我要求，允许自己偶尔卡壳
3. 提前告知听众自己有口吃，减轻心理压力
4. 正念冥想可以帮助缓解说话前的紧张
5. 寻求专业心理咨询，处理深层焦虑""",
        "metadata": {"category": "心理调节", "level": "通用", "type": "心理指导"}
    },
    {
        "id": "practice_001",
        "content": """日常练习计划：
1. 每天早晨进行10分钟呼吸训练
2. 朗读练习20分钟，使用慢速轻柔的方式
3. 对着镜子练习，观察自己的口型和表情
4. 录音回听，分析自己的进步
5. 参加口吃互助小组，互相鼓励""",
        "metadata": {"category": "练习计划", "level": "通用", "type": "计划安排"}
    },
    {
        "id": "difficult_sounds_001",
        "content": """难发音处理方法：
1. 找出自己最容易卡壳的音节
2. 针对这些音节进行单独练习
3. 使用替代词策略，暂时避开难发音
4. 用唱歌的方式说难发的词
5. 从轻到重，逐步增加难度""",
        "metadata": {"category": "难发音", "level": "中级", "type": "应对策略"}
    },
    {
        "id": "children_001",
        "content": """儿童口吃家长指南：
1. 不要打断或催促孩子说话
2. 降低家庭沟通压力，营造轻松氛围
3. 避免模仿孩子的口吃行为
4. 多给予正面鼓励，少批评
5. 如果持续6个月以上，建议寻求专业帮助""",
        "metadata": {"category": "儿童口吃", "level": "家长指导", "type": "指导手册"}
    },
    {
        "id": "fluency_001",
        "content": """流畅性强化训练：
1. 从单字开始，逐步过渡到句子
2. 使用延音技巧，把元音适当拉长
3. 在词与词之间加入轻微停顿
4. 保持呼吸平稳，不要憋气说话
5. 每天坚持练习，建立新的说话模式""",
        "metadata": {"category": "流畅性", "level": "高级", "type": "强化训练"}
    }
]


def init_knowledge_base():
    """初始化知识库"""
    print("=" * 60)
    print("📚 初始化口吃矫正知识库")
    print("=" * 60)
    
    # 创建集合
    collection = chroma_client.get_or_create_collection(
        name="stutter_correction",
        metadata={"description": "口吃矫正知识库", "created_by": "AI Service"}
    )
    
    print(f"✅ 集合 'stutter_correction' 创建/获取成功")
    
    # 准备数据
    documents = [item["content"] for item in STUTTER_CORRECTION_MATERIALS]
    ids = [item["id"] for item in STUTTER_CORRECTION_MATERIALS]
    metadatas = [item["metadata"] for item in STUTTER_CORRECTION_MATERIALS]
    
    print(f"📄 准备导入 {len(documents)} 条文档...")
    
    # 生成 embeddings
    print("🔄 正在生成向量嵌入...")
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=documents
    )
    embeddings = [item.embedding for item in response.data]
    print(f"✅ 生成 {len(embeddings)} 个向量，维度: {len(embeddings[0])}")
    
    # 添加到 ChromaDB
    print("💾 正在保存到 ChromaDB...")
    collection.add(
        embeddings=embeddings,
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    
    print("✅ 知识库初始化完成！\n")
    
    # 显示统计
    count = collection.count()
    print(f"📊 当前知识库文档数: {count}")
    
    # 显示所有文档
    print("\n📋 已导入文档列表:")
    for i, item in enumerate(STUTTER_CORRECTION_MATERIALS, 1):
        meta = item["metadata"]
        print(f"  {i}. [{meta['category']}] {item['id']}")


def test_search():
    """测试搜索功能"""
    print("\n" + "=" * 60)
    print("🔍 测试知识库搜索")
    print("=" * 60)
    
    collection = chroma_client.get_collection("stutter_correction")
    
    # 测试查询
    queries = [
        "怎么练习呼吸",
        "孩子口吃怎么办",
        "说话快怎么控制",
    ]
    
    for query in queries:
        print(f"\n🔍 查询: '{query}'")
        print("-" * 40)
        
        results = collection.query(
            query_texts=[query],
            n_results=2,
            include=["documents", "metadatas", "distances"]
        )
        
        for i in range(len(results['ids'][0])):
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            print(f"  结果 {i+1}:")
            print(f"    类别: {meta['category']}")
            print(f"    相似度: {1 - distance:.4f}")
            print(f"    内容: {doc[:80]}...")


if __name__ == "__main__":
    # 检查 API Key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ 错误: 未设置 DEEPSEEK_API_KEY 环境变量")
        print("请先设置: export DEEPSEEK_API_KEY='your-api-key'")
        exit(1)
    
    # 运行初始化
    init_knowledge_base()
    
    # 测试搜索
    test_search()
    
    print("\n" + "=" * 60)
    print("✅ 知识库搭建完成！")
    print("=" * 60)
    print("\n现在你可以:")
    print("1. 启动服务: python main.py")
    print("2. 通过 API 进行搜索: POST /knowledge-base/search")
    print("3. Go 后端可以调用这个服务来查询口吃矫正知识")
