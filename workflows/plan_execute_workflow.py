"""
Plan & Execute 工作流 - 用于复杂长期任务
支持流式输出，展示 AI 的规划和执行过程
"""

import json
import os
from typing import TypedDict, List, Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ============ 状态定义 ============

class PlanExecuteState(TypedDict):
    # 基础信息
    thread_id: str
    user_id: str
    user_message: str
    user_profile: Dict[str, Any]
    
    # Plan & Execute 核心字段
    plan: List[str]         # AI 生成的步骤列表
    results: List[str]      # 每步执行结果
    current_step: int       # 当前执行到第几步
    total_steps: int        # 总共几步
    
    # 流式回调
    stream_callback: Optional[Callable]  # 用于向前端发送进度
    
    # 最终输出
    final_response: str
    execution_path: List[str]
    status: str
    
    # 内部字段，用于状态传递
    task_context_id: Optional[str]  # 任务ID，用于获取回调


# ============ LLM 调用 ============

def call_doubao(prompt: str, system_prompt: str = "你是一个专业的口吃矫正专家。", expect_json: bool = False):
    """调用豆包 API"""
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    
    model = os.getenv("DOUBAO_MODEL_ID", "ep-m-20260113142855-wqkg9")
    api_key = os.getenv("DOUBAO_API_KEY", "")
    
    if not api_key:
        print("⚠️  未配置 API Key，使用 Mock 模式")
        if expect_json:
            return {
                "steps": [
                    "评估当前口吃严重程度和类型",
                    "制定第一个月基础训练计划",
                    "制定第二个月进阶计划",
                    "制定第三个月巩固计划"
                ]
            }
        return "【Mock回复】这是模拟的执行结果"
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3"
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.3 if expect_json else 0.7,
    )
    
    content = response.choices[0].message.content
    
    if expect_json:
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except:
            return {"steps": ["分析需求", "制定方案", "执行计划"]}
    
    return content


# ============ 节点函数 ============

def planner(state: PlanExecuteState) -> Dict:
    """
    规划节点：AI 把大任务拆解成具体步骤
    """
    print(f"\n📋 [Planner] 拆解任务：{state['user_message']}")
    
    # 从全局变量获取回调
    task_id = state.get("task_context_id")
    callback = _get_task_callback(task_id) if task_id else None
    
    # 流式通知：开始规划
    if callback:
        callback({
            "type": "planning",
            "message": "正在分析您的需求并制定执行计划..."
        })
    
    user_profile = state.get('user_profile', {})
    severity = user_profile.get('severity_level', 3)
    stuttering_type = user_profile.get('stuttering_type', '未知')
    
    prompt = f"""你是口吃矫正专家。用户有一个需要多步骤完成的复杂目标，请帮他拆解成3-5个具体的执行步骤。

用户目标：{state['user_message']}
用户情况：严重程度 {severity}/5，口吃类型：{stuttering_type}

要求：
- 每个步骤具体、可执行
- 步骤之间有逻辑顺序
- 符合用户当前水平
- 用简洁的语言描述

输出格式（严格JSON）：
{{
    "steps": [
        "步骤1：...",
        "步骤2：...",
        "步骤3：..."
    ]
}}"""
    
    result = call_doubao(prompt, expect_json=True)
    steps = result.get("steps", [])
    
    print(f"📋 [Planner] 拆解完成，共 {len(steps)} 步")
    
    # 流式通知：计划已生成
    if callback:
        callback({
            "type": "plan",
            "steps": steps,
            "total_steps": len(steps)
        })
    
    return {
        "plan": steps,
        "total_steps": len(steps),
        "current_step": 0,
        "results": [],
        "execution_path": ["planner"]
    }


def executor(state: PlanExecuteState) -> Dict:
    """
    执行节点：执行当前这一步
    """
    current_step = state["current_step"]
    total_steps = state["total_steps"]
    current_task = state["plan"][current_step]
    
    print(f"\n⚙️  [Executor] 执行第 {current_step + 1}/{total_steps} 步：{current_task}")
    
    # 从全局变量获取回调
    task_id = state.get("task_context_id")
    callback = _get_task_callback(task_id) if task_id else None
    
    # 流式通知：开始执行当前步骤
    if callback:
        callback({
            "type": "step_start",
            "current": current_step + 1,
            "total": total_steps,
            "description": current_task
        })
    
    # 收集前面步骤的结果作为上下文
    previous_results = ""
    if state["results"]:
        previous_results = "\n".join([
            f"步骤{i+1}已完成：{r[:150]}..."
            for i, r in enumerate(state["results"])
        ])
    
    user_profile = state.get('user_profile', {})
    
    prompt = f"""你是口吃矫正专家。请完成当前这个具体任务。

用户总目标：{state['user_message']}
用户画像：严重程度 {user_profile.get('severity_level', 3)}/5，类型：{user_profile.get('stuttering_type', '未知')}
优势：{', '.join(user_profile.get('strengths', [])) or '暂无'}
弱项：{', '.join(user_profile.get('weaknesses', [])) or '暂无'}

当前任务（第{current_step + 1}步，共{total_steps}步）：
{current_task}

{previous_results if previous_results else "这是第一步，请详细展开。"}

请针对当前任务给出详细、具体、可操作的内容。如果涉及训练计划，请包含：
1. 具体练习方法
2. 每日练习时长
3. 注意事项
4. 预期效果"""

    result = call_doubao(prompt)
    
    print(f"⚙️  [Executor] 第 {current_step + 1} 步完成")
    
    # 流式通知：步骤完成
    if callback:
        callback({
            "type": "step_complete",
            "current": current_step + 1,
            "total": total_steps,
            "result": result[:200] + "..." if len(result) > 200 else result
        })
    
    return {
        "results": state["results"] + [result],
        "current_step": current_step + 1,
        "execution_path": state.get("execution_path", []) + [f"executor_step_{current_step + 1}"]
    }


def should_continue(state: PlanExecuteState) -> str:
    """
    判断是否还有任务要执行
    """
    if state["current_step"] < state["total_steps"]:
        return "executor"
    else:
        return "aggregator"


def aggregator(state: PlanExecuteState) -> Dict:
    """
    汇总节点：把所有步骤整合成完整回复
    """
    print(f"\n📝 [Aggregator] 汇总 {len(state['results'])} 个步骤")
    
    # 从全局变量获取回调
    task_id = state.get("task_context_id")
    callback = _get_task_callback(task_id) if task_id else None
    
    # 流式通知：开始汇总
    if callback:
        callback({
            "type": "aggregating",
            "message": "正在整合所有内容，生成完整方案..."
        })
    
    # 构建步骤和结果的对应
    steps_and_results = ""
    for i, (step, result) in enumerate(zip(state["plan"], state["results"])):
        steps_and_results += f"\n{'='*50}\n【步骤{i+1}】{step}\n{'='*50}\n{result}\n"
    
    user_profile = state.get('user_profile', {})
    
    prompt = f"""请将以下内容整合成一份完整、专业、温暖的回复。

用户原始需求：{state['user_message']}
用户情况：严重程度 {user_profile.get('severity_level', 3)}/5

各步骤详细内容：
{steps_and_results}

整合要求：
1. 开头要有温度，认可用户的决心
2. 按时间线或逻辑线组织，不要简单拼接
3. 突出各阶段的重点和衔接
4. 给出可执行的行动建议
5. 结尾鼓励用户，强调坚持的重要性
6. 使用 emoji 增加可读性
7. 总字数控制在 800 字以内"""

    final = call_doubao(prompt)
    
    print(f"✅ [Aggregator] 汇总完成")
    
    # 流式通知：完成
    if callback:
        callback({
            "type": "final",
            "content": final
        })
    
    return {
        "final_response": final,
        "execution_path": state.get("execution_path", []) + ["aggregator"],
        "status": "completed"
    }


# ============ 构建工作流 ============

def create_plan_execute_workflow():
    """
    Plan & Execute 工作流
    
    流程：
    START → planner (AI 拆解任务)
                ↓
            executor (执行当前步)
                ↓
        should_continue 判断
          ↙            ↘
    还有任务         没任务了
    回到executor     去aggregator
                         ↓
                        END
    """
    workflow = StateGraph(PlanExecuteState)
    
    # 添加节点
    workflow.add_node("planner", planner)
    workflow.add_node("executor", executor)
    workflow.add_node("aggregator", aggregator)
    
    # 入口
    workflow.set_entry_point("planner")
    
    # planner -> executor
    workflow.add_edge("planner", "executor")
    
    # executor -> 判断
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "executor": "executor",
            "aggregator": "aggregator"
        }
    )
    
    # aggregator -> END
    workflow.add_edge("aggregator", END)
    
    # 内存检查点（支持流式）
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ============ 任务状态管理（用于轮询） ============

from datetime import datetime, timedelta

class TaskManager:
    """管理 Plan & Execute 任务状态"""
    
    def __init__(self):
        self.tasks = {}  # task_id -> task_info
        self.results = {}  # task_id -> result
    
    def create_task(self, thread_id: str, user_id: str, user_message: str) -> str:
        """创建新任务"""
        task_id = thread_id
        self.tasks[task_id] = {
            "task_id": task_id,
            "user_id": user_id,
            "user_message": user_message,
            "status": "planning",  # planning, executing, aggregating, completed, failed
            "plan": [],
            "current_step": 0,
            "total_steps": 0,
            "step_results": [],
            "execution_path": [],
            "final_response": None,
            "error": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        return task_id
    
    def update_task(self, task_id: str, **kwargs):
        """更新任务状态"""
        if task_id in self.tasks:
            self.tasks[task_id].update(kwargs)
            self.tasks[task_id]["updated_at"] = datetime.now()
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        return self.tasks.get(task_id)
    
    def cleanup_old_tasks(self, hours: int = 24):
        """清理过期任务"""
        cutoff = datetime.now() - timedelta(hours=hours)
        expired = [tid for tid, task in self.tasks.items() 
                   if task["created_at"] < cutoff]
        for tid in expired:
            del self.tasks[tid]
            if tid in self.results:
                del self.results[tid]

# 全局任务管理器
task_manager = TaskManager()

# 全局回调存储（用于在工作流节点中访问）
_task_callbacks: Dict[str, Callable] = {}


def _get_task_callback(task_id: str) -> Optional[Callable]:
    """获取指定任务的回调函数"""
    return _task_callbacks.get(task_id)


def _set_task_callback(task_id: str, callback: Callable):
    """设置指定任务的回调函数"""
    _task_callbacks[task_id] = callback


def _clear_task_callback(task_id: str):
    """清除指定任务的回调函数"""
    if task_id in _task_callbacks:
        del _task_callbacks[task_id]


# ============ 带状态更新的工作流 ============

def run_plan_execute_with_updates(task_id: str, initial_state: Dict):
    """运行工作流并实时更新状态"""
    
    def stream_callback(data: Dict):
        """更新任务进度"""
        msg_type = data.get("type")
        
        if msg_type == "plan":
            task_manager.update_task(
                task_id,
                status="executing",
                plan=data.get("steps", []),
                total_steps=data.get("total_steps", 0)
            )
        elif msg_type == "step_start":
            task_manager.update_task(
                task_id,
                current_step=data.get("current", 0),
                status="executing"
            )
        elif msg_type == "step_complete":
            task = task_manager.get_task(task_id)
            if task:
                new_results = task.get("step_results", []) + [data.get("result", "")]
                task_manager.update_task(
                    task_id,
                    step_results=new_results,
                    current_step=data.get("current", 0)
                )
        elif msg_type == "aggregating":
            task_manager.update_task(task_id, status="aggregating")
        elif msg_type == "final":
            task_manager.update_task(
                task_id,
                status="completed",
                final_response=data.get("content", "")
            )
    
    # 注册回调到全局存储
    _set_task_callback(task_id, stream_callback)
    
    try:
        workflow = get_plan_execute_workflow()
        # 创建不包含回调的状态副本给工作流使用
        workflow_state = {k: v for k, v in initial_state.items() if k != "stream_callback"}
        # 添加 task_id 到状态，以便节点函数可以获取回调
        workflow_state["task_context_id"] = task_id
        result = workflow.invoke(
            workflow_state,
            config={"configurable": {"thread_id": task_id}}
        )
        
        # 更新最终结果
        task_manager.update_task(
            task_id,
            status="completed",
            final_response=result.get("final_response", ""),
            execution_path=result.get("execution_path", []),
            plan=result.get("plan", []),
            step_results=result.get("results", [])
        )
        
        return result
        
    except Exception as e:
        task_manager.update_task(
            task_id,
            status="failed",
            error=str(e)
        )
        raise
    finally:
        # 清理回调，避免内存泄漏
        _clear_task_callback(task_id)


# ============ 单例 ============

_plan_execute_workflow = None

def get_plan_execute_workflow():
    global _plan_execute_workflow
    if _plan_execute_workflow is None:
        _plan_execute_workflow = create_plan_execute_workflow()
    return _plan_execute_workflow
