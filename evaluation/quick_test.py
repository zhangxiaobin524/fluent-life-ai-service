"""
快速测试脚本 - 无需真实 workflow 即可体验评测功能
运行方式: python -m evaluation.quick_test
"""

import os
from evaluation import AIEvaluator, TEST_CASES


def mock_expert_team(test_case: dict) -> dict:
    """
    ExpertTeam Mock 实现
    模拟不同质量的输出用于测试评测系统
    """
    expected = test_case.get("expected", {})
    question_type = expected.get("question_type", "general")
    
    # 根据测试用例 ID 决定模拟质量（奇数好，偶数差）
    is_good = int(test_case["id"].split("_")[1]) % 2 == 1
    
    if is_good:
        # 高质量输出
        return {
            "question_type": question_type,
            "complexity": expected.get("complexity", "medium"),
            "routing_reason": f"用户表达了{question_type}相关需求，语义分析匹配",
            "experts_involved": expected.get("should_involve", ["通用专家"]),
            "execution_path": ["router", f"{question_type}_expert", "aggregator"],
            "final_response": f"针对您的{question_type}需求，我建议：1)详细分析... 2)具体方案... 3)后续跟进...",
            "status": "completed"
        }
    else:
        # 低质量输出（用于测试低分检测）
        return {
            "question_type": "general",  # 错误：未正确识别
            "complexity": "simple",
            "routing_reason": "默认路由",
            "experts_involved": ["通用专家"],  # 错误：未调用专业专家
            "execution_path": ["router"],
            "final_response": "好的，我明白了。",
            "status": "completed"
        }


def mock_training_plan(test_case: dict) -> dict:
    """
    TrainingPlan Mock 实现
    """
    profile = test_case.get("user_profile", {})
    expected = test_case.get("expected", {})
    is_good = int(test_case["id"].split("_")[1]) % 2 == 1
    
    level = profile.get("level", "初级")
    available_time = profile.get("available_time", "30分钟/天")
    
    if is_good:
        # 高质量计划
        duration = expected.get("duration_days", (7, 14))
        return {
            "title": f"个性化{level}口吃矫正训练计划",
            "duration_days": duration[0] if isinstance(duration, tuple) else duration,
            "daily_time": available_time,
            "difficulty": profile.get("level", "初级"),
            "exercises": expected.get("should_include", ["基础训练"]),
            "goals": [
                f"第1阶段：掌握{profile.get('recommended_techniques', ['基础技巧'])[0]}",
                "第2阶段：场景应用",
                "第3阶段：巩固提升"
            ],
            "personalization_note": f"针对您的{profile.get('stuttering_type', '口吃')}定制",
            "status": "completed"
        }
    else:
        # 低质量计划（通用模板）
        return {
            "title": "标准训练计划",
            "duration_days": 7,
            "daily_time": "1小时",
            "difficulty": "通用",
            "exercises": ["练习1", "练习2", "练习3"],  # 无针对性
            "goals": ["改善口吃"],
            "personalization_note": None,
            "status": "completed"
        }


def main():
    """运行快速测试"""
    
    print("="*70)
    print(" LLM-as-a-Judge 快速测试")
    print(" 使用 Mock AI 输出测试评测系统")
    print("="*70)
    
    # 检查 API Key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("\n⚠️  请先设置 DEEPSEEK_API_KEY 环境变量")
        print("   export DEEPSEEK_API_KEY=sk-xxxxxx")
        return
    
    evaluator = AIEvaluator(api_key)
    
    # 测试 ExpertTeam（只测前3个用例，节省 API 费用）
    print("\n🧪 测试 ExpertTeam 模块 (3个用例)...")
    expert_cases = TEST_CASES["expert_team"][:3]
    expert_result = evaluator.evaluate_module("expert_team", expert_cases, mock_expert_team)
    
    print(f"\n   ExpertTeam 得分: {expert_result['total_score']}/10.0")
    for dim, stats in expert_result['dimensions'].items():
        print(f"   - {dim}: {stats['avg_score']}")
    
    # 测试 TrainingPlan（只测前3个用例）
    print("\n🧪 测试 TrainingPlan 模块 (3个用例)...")
    plan_cases = TEST_CASES["training_plan"][:3]
    plan_result = evaluator.evaluate_module("training_plan", plan_cases, mock_training_plan)
    
    print(f"\n   TrainingPlan 得分: {plan_result['total_score']}/10.0")
    for dim, stats in plan_result['dimensions'].items():
        print(f"   - {dim}: {stats['avg_score']}")
    
    # 保存报告
    evaluator.generate_report("quick_test_report.md")
    
    print("\n" + "="*70)
    print(" 测试完成！")
    print(" 报告已保存: quick_test_report.md")
    print("\n 说明:")
    print(" - 奇数ID用例模拟高质量输出，应该得高分")
    print(" - 偶数ID用例模拟低质量输出，应该得低分")
    print(" - 如果评分符合这个规律，说明评测系统工作正常")
    print("="*70)


if __name__ == "__main__":
    main()
