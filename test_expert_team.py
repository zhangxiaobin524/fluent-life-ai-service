"""
专家团队工作流测试脚本
"""
import requests
import json

BASE_URL = "http://localhost:8002"


def test_expert_team(message: str, user_id: str = "test_user_001"):
    """测试专家团队模式"""
    print(f"\n{'='*60}")
    print(f"📝 用户问题: {message}")
    print('='*60)
    
    response = requests.post(
        f"{BASE_URL}/workflow/expert-team/chat",
        json={
            "user_id": user_id,
            "user_message": message,
            "user_profile": {
                "stutter_type": "重复型",
                "severity": "中度",
                "training_summary": "最近训练5次，冥想3次"
            }
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n📊 路由信息:")
        print(f"   - 问题类型: {result['question_type']}")
        print(f"   - 复杂度: {result['complexity']}")
        print(f"   - 参与专家: {', '.join(result['experts_involved'])}")
        print(f"   - 执行路径: {' -> '.join(result['execution_path'])}")
        print(f"\n💬 AI回复:\n{result['final_response']}")
    else:
        print(f"❌ 错误: {response.text}")


def test_all_scenarios():
    """测试各种场景"""
    
    # 场景1: 简单问题（应该直接汇总）
    test_expert_team("你好", "user_001")
    
    # 场景2: 诊断类问题（应该走诊断专家）
    test_expert_team("我最近说话总是卡在第一字，怎么回事？", "user_002")
    
    # 场景3: 心理问题（应该直接走心理支持）
    test_expert_team("我好沮丧，练习了一个月没效果", "user_003")
    
    # 场景4: 数据分析类
    test_expert_team("帮我分析一下最近的训练数据", "user_004")
    
    # 场景5: 训练计划类
    test_expert_team("给我制定一个下周的训练计划", "user_005")


if __name__ == "__main__":
    print("🚀 专家团队工作流测试")
    print("确保 AI 服务已启动: uvicorn main:app --port 8002")
    
    # 测试单个场景
    test_expert_team("我最近说话总是卡在第一字，怎么回事？")
    
    # 或者测试全部场景
    # test_all_scenarios()
