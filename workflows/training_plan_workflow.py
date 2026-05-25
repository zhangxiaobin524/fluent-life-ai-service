"""
智能训练计划生成工作流 - 基于 LangGraph
支持多轮对话式计划定制
"""

import json
import uuid
from typing import TypedDict, Annotated, Literal, Optional, List, Dict, Any
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
import operator


# ============ 状态定义 ============

class TrainingPlanState(TypedDict):
    """工作流状态"""
    # 基础信息
    thread_id: str
    user_id: str
    user_profile: Dict[str, Any]
    training_history: List[Dict]
    
    # 计划生成
    current_plan: Optional[Dict[str, Any]]
    plan_content: Optional[str]  # 用于展示的计划文本
    
    # 用户反馈
    feedback: Optional[Literal["too_easy", "too_hard", "adjust_focus", "good"]]
    feedback_detail: Optional[str]  # 用户的详细说明
    
    # 控制字段
    adjustment_count: int
    max_adjustments: int
    status: Literal["running", "completed", "error"]
    
    # 对话历史
    conversation_history: Annotated[List[Dict], operator.add]
    
    # 输出
    final_plan: Optional[Dict[str, Any]]
    final_plan_id: Optional[str]


# ============ LLM 配置 ============

def call_llm(prompt: str, system_prompt: str = "你是一个专业的口吃矫正训练师。", expect_json: bool = False) -> str:
    """调用 DeepSeek API"""
    import os
    from openai import OpenAI
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        # Mock 模式
        print("⚠️  未配置 DEEPSEEK_API_KEY，使用 Mock 模式")
        if expect_json:
            return '{"title": "7天口吃矫正计划", "description": "Mock计划", "difficulty": "easy", "daily_plans": [], "summary": "请配置API Key"}'
        return "【Mock回复】请配置 DEEPSEEK_API_KEY 环境变量以使用真实AI功能。"
    
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.3 if expect_json else 0.7,
    )
    
    return response.choices[0].message.content


# ============ 节点函数 ============

def analyze_user(state: TrainingPlanState) -> TrainingPlanState:
    """分析用户数据，生成初步计划"""
    user_profile = state["user_profile"]
    training_history = state.get("training_history", [])
    
    # 构建提示词
    prompt = f"""基于以下用户信息，生成一个7天的口吃矫正训练计划。

用户画像：
- 口吃类型：{user_profile.get('stuttering_type', '未知')}
- 严重程度：{user_profile.get('severity_level', 5)}/10
- 触发因素：{', '.join(user_profile.get('trigger_factors', []))}
- 推荐技巧：{', '.join(user_profile.get('recommended_techniques', []))}

训练历史摘要：
{json.dumps(training_history[-5:] if training_history else [], ensure_ascii=False, indent=2)}

请生成一个详细的7天训练计划，包含：
1. 每天的主要训练类型（冥想/气流/脱敏/实战）
2. 每天的预计时长
3. 训练重点和目标
4. 难度说明

请以JSON格式返回：
{{
    "title": "计划标题",
    "description": "计划描述",
    "difficulty": "easy/medium/hard",
    "daily_plans": [
        {{
            "day": 1,
            "type": "meditation/airflow/exposure/practice",
            "title": "训练标题",
            "duration": 30,
            "focus": "训练重点",
            "goals": ["目标1", "目标2"]
        }}
    ],
    "summary": "计划总结"
}}"""

    try:
        content = call_llm(prompt, expect_json=True)
        
        if not content:
            raise ValueError("AI 返回内容为空")
        
        # 提取 JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        plan = json.loads(content.strip())
        
        if not plan or not isinstance(plan, dict):
            raise ValueError("解析后的计划为空")
        
        # 生成展示文本
        plan_content = format_plan_for_display(plan)
        
        return {
            **state,
            "current_plan": plan,
            "plan_content": plan_content,
            "adjustment_count": 0,
            "conversation_history": [{
                "role": "ai",
                "content": f"已为您生成初步训练计划：{plan.get('title', '自定义训练计划')}",
                "timestamp": datetime.now().isoformat()
            }]
        }
    except Exception as e:
        import traceback
        error_msg = f"生成计划时出错：{str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {
            **state,
            "status": "error",
            "conversation_history": [{
                "role": "ai",
                "content": f"生成计划时出错：{str(e)}",
                "timestamp": datetime.now().isoformat()
            }]
        }


def format_plan_for_display(plan: Dict) -> str:
    """格式化计划用于展示"""
    lines = [
        f"## {plan['title']}",
        f"\n{plan['description']}",
        f"\n**难度**：{plan['difficulty']}",
        "\n### 7天安排：",
    ]
    
    for day_plan in plan.get("daily_plans", []):
        lines.append(f"\n**第{day_plan['day']}天** - {day_plan['title']}")
        lines.append(f"- 类型：{day_plan['type']}")
        lines.append(f"- 时长：{day_plan['duration']}分钟")
        lines.append(f"- 重点：{day_plan['focus']}")
    
    lines.append(f"\n### 总结\n{plan['summary']}")
    return "\n".join(lines)


def present_plan(state: TrainingPlanState) -> TrainingPlanState:
    """向用户展示计划并询问反馈"""
    plan = state["current_plan"]
    adjustment_count = state["adjustment_count"]
    
    if adjustment_count == 0:
        message = f"我为您制定了训练计划【{plan['title']}】，难度为{plan['difficulty']}。\n\n这个计划如何？"
    else:
        message = f"已根据您的反馈调整（第{adjustment_count}次），难度现在是{plan['difficulty']}。\n\n现在满意吗？"
    
    return {
        **state,
        "conversation_history": [{
            "role": "ai",
            "content": message,
            "timestamp": datetime.now().isoformat()
        }]
    }


def process_feedback(state: TrainingPlanState) -> TrainingPlanState:
    """处理用户反馈，调整计划"""
    feedback = state["feedback"]
    current_plan = state["current_plan"]
    feedback_detail = state.get("feedback_detail", "")
    adjustment_count = state["adjustment_count"]
    
    if feedback == "good":
        # 用户满意，结束流程
        return {
            **state,
            "final_plan": current_plan,
            "status": "completed",
            "conversation_history": [{
                "role": "ai",
                "content": "太好了！计划已确认并保存。祝您训练顺利！",
                "timestamp": datetime.now().isoformat()
            }]
        }
    
    # 需要调整
    prompt = f"""基于用户反馈，调整以下训练计划。

当前计划：
{json.dumps(current_plan, ensure_ascii=False, indent=2)}

用户反馈：{feedback}
详细说明：{feedback_detail}

调整要求：
- 如果反馈是"太简单"：增加训练难度、延长时长、增加高级技巧
- 如果反馈是"有点难"：降低难度、缩短时长、增加基础练习
- 如果反馈是"调整重点"：根据用户说明调整训练重点

请直接返回调整后的完整计划（JSON格式），保持相同结构。"""

    try:
        content = call_llm(prompt)
        
        # 提取 JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        new_plan = json.loads(content.strip())
        plan_content = format_plan_for_display(new_plan)
        
        adjustment_messages = {
            "too_easy": "收到，我增加了训练难度和时长。",
            "too_hard": "明白，我降低了难度，让计划更容易坚持。",
            "adjust_focus": "好的，我根据您的需求调整了训练重点。"
        }
        
        return {
            **state,
            "current_plan": new_plan,
            "plan_content": plan_content,
            "adjustment_count": adjustment_count + 1,
            "conversation_history": [{
                "role": "ai",
                "content": adjustment_messages.get(feedback, "已调整计划。"),
                "timestamp": datetime.now().isoformat()
            }]
        }
    except Exception as e:
        return {
            **state,
            "status": "error",
            "conversation_history": [{
                "role": "ai",
                "content": f"调整计划时出错：{str(e)}",
                "timestamp": datetime.now().isoformat()
            }]
        }


def check_should_continue(state: TrainingPlanState) -> Literal["present", "end"]:
    """检查是否应该继续调整"""
    # 如果已完成，结束
    if state["status"] == "completed":
        return "end"
    
    # 如果出错，结束
    if state["status"] == "error":
        return "end"
    
    # 如果调整次数超过最大限制，结束
    if state["adjustment_count"] >= state["max_adjustments"]:
        return "end"
    
    # 继续展示
    return "present"


def check_has_feedback(state: TrainingPlanState) -> Literal["process", "wait"]:
    """检查是否有用户反馈，没有则等待"""
    if state.get("feedback") is None:
        return "wait"  # 没有反馈，结束等待用户
    return "process"  # 有反馈，继续处理


# ============ 构建工作流 ============

def create_training_plan_workflow():
    """创建训练计划工作流"""
    
    # 定义状态图
    workflow = StateGraph(TrainingPlanState)
    
    # 添加节点
    workflow.add_node("analyze", analyze_user)
    workflow.add_node("present", present_plan)
    workflow.add_node("process_feedback", process_feedback)
    
    # 设置入口
    workflow.set_entry_point("analyze")
    
    # 添加边
    workflow.add_edge("analyze", "present")
    
    # 条件边：present 后检查是否有反馈
    workflow.add_conditional_edges(
        "present",
        check_has_feedback,
        {
            "process": "process_feedback",  # 有反馈，继续处理
            "wait": END                      # 没反馈，结束等待用户
        }
    )
    
    # 条件边：处理完反馈后，检查是否继续
    workflow.add_conditional_edges(
        "process_feedback",
        check_should_continue,
        {
            "present": "present",  # 继续展示（用户不满意）
            "end": END             # 结束（用户说 good 或超次数）
        }
    )
    
    # 添加内存检查点（支持中断/恢复）
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)


# ============ 单例 ============

_workflow = None

def get_workflow():
    """获取工作流实例（单例）"""
    global _workflow
    if _workflow is None:
        _workflow = create_training_plan_workflow()
    return _workflow
