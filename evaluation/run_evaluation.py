"""
评测执行脚本
运行方式: python -m evaluation.run_evaluation
"""

import os
import sys
import asyncio
from typing import Dict, Any

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import AIEvaluator, TEST_CASES


# ========================================
# ExpertTeam 适配器
# ========================================

async def call_expert_team_workflow(test_case: Dict) -> Dict:
    """
    调用 ExpertTeam 工作流
    将测试用例转换为工作流输入
    """
    from workflows.expert_team_workflow import get_expert_workflow, ExpertTeamState
    
    workflow = get_expert_workflow()
    
    # 构建初始状态
    initial_state: ExpertTeamState = {
        "thread_id": f"test_{test_case['id']}",
        "user_id": "eval_user_001",
        "user_message": test_case["user_message"],
        "user_profile": test_case.get("user_profile", {}),
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
    
    try:
        # 运行工作流（需要传入 config 以支持 checkpointer）
        config = {"configurable": {"thread_id": initial_state["thread_id"]}}
        result = await workflow.ainvoke(initial_state, config)
        
        return {
            "question_type": result.get("question_type", "unknown"),
            "complexity": result.get("complexity", "simple"),
            "routing_reason": result.get("routing_reason", ""),
            "experts_involved": result.get("experts_involved", []),
            "execution_path": result.get("execution_path", []),
            "final_response": result.get("final_response", "")[:500],  # 截断避免过长
            "diagnosis_result": result.get("diagnosis_result") is not None,
            "data_analysis_result": result.get("data_analysis_result") is not None,
            "plan_result": result.get("plan_result") is not None,
            "support_result": result.get("support_result") is not None,
        }
    except Exception as e:
        # 如果 workflow 运行失败，返回错误信息
        return {
            "error": str(e),
            "question_type": "error",
            "experts_involved": [],
            "final_response": f"工作流执行失败: {e}"
        }


async def call_expert_team_async(test_case: Dict) -> Dict:
    """异步调用"""
    return await call_expert_team_workflow(test_case)


# ========================================
# TrainingPlan 适配器
# ========================================

async def call_training_plan_workflow(test_case: Dict) -> Dict:
    """
    调用 TrainingPlan 工作流
    """
    from workflows.training_plan_workflow import get_workflow, TrainingPlanState
    
    workflow = get_workflow()
    
    initial_state: TrainingPlanState = {
        "thread_id": f"test_{test_case['id']}",
        "user_id": "eval_user_001",
        "user_profile": test_case.get("user_profile", {}),
        "training_history": test_case.get("training_history", []),
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
    
    try:
        # 运行工作流（需要传入 config 以支持 checkpointer）
        config = {"configurable": {"thread_id": initial_state["thread_id"]}}
        result = await workflow.ainvoke(initial_state, config)
        
        final_plan = result.get("final_plan", {})
        
        return {
            "has_final_plan": final_plan is not None,
            "plan_title": final_plan.get("title", "") if final_plan else "",
            "duration_days": final_plan.get("duration_days", 0) if final_plan else 0,
            "daily_time": final_plan.get("daily_time", "") if final_plan else "",
            "exercises": final_plan.get("exercises", []) if final_plan else [],
            "difficulty": final_plan.get("difficulty", "") if final_plan else "",
            "goals": final_plan.get("goals", []) if final_plan else [],
            "plan_content_preview": result.get("plan_content", "")[:300] if result.get("plan_content") else "",
            "adjustment_count": result.get("adjustment_count", 0),
            "status": result.get("status", "unknown")
        }
    except Exception as e:
        return {
            "error": str(e),
            "has_final_plan": False,
            "plan_title": "生成失败",
            "status": "error"
        }


async def call_training_plan_async(test_case: Dict) -> Dict:
    """异步调用"""
    return await call_training_plan_workflow(test_case)


# ========================================
# 主运行函数
# ========================================

async def main():
    """运行完整评测"""
    
    print("="*70)
    print(" Fluent Life AI 模块质量评测 (LLM-as-a-Judge)")
    print("="*70)
    
    # 检查环境变量
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("\n⚠️  警告: 未设置 DEEPSEEK_API_KEY 环境变量")
        print("   评测功能将无法使用")
        print("   请设置: export DEEPSEEK_API_KEY=sk-xxxxxx")
        return
    
    evaluator = AIEvaluator(api_key)
    
    # 评测 ExpertTeam 模块
    print("\n📋 开始评测 ExpertTeam 模块...")
    expert_results = await evaluator.evaluate_module_async(
        "expert_team",
        TEST_CASES["expert_team"],
        call_expert_team_async
    )
    
    # 评测 TrainingPlan 模块
    print("\n📋 开始评测 TrainingPlan 模块...")
    plan_results = await evaluator.evaluate_module_async(
        "training_plan",
        TEST_CASES["training_plan"],
        call_training_plan_async
    )
    
    # 生成报告
    print("\n📊 生成评测报告...")
    report = evaluator.generate_report("evaluation_report.md")
    
    # 控制台摘要
    print("\n" + "="*70)
    print(" 评测结果摘要")
    print("="*70)
    print(f"\nExpertTeam 模块: {expert_results['total_score']}/10.0 ({expert_results['grade']})")
    print(f"TrainingPlan 模块: {plan_results['total_score']}/10.0 ({plan_results['grade']})")
    
    avg_score = (expert_results['total_score'] + plan_results['total_score']) / 2
    print(f"\n综合评分: {avg_score:.2f}/10.0")
    
    if avg_score >= 8.0:
        print("✅ 整体表现良好")
    elif avg_score >= 6.0:
        print("⚠️  整体表现一般，有改进空间")
    else:
        print("❌ 整体表现较差，需要重点优化")
    
    print(f"\n详细报告已保存: evaluation_report.md")


if __name__ == "__main__":
    asyncio.run(main())
