"""
LangGraph 异步多 Agent 面试工作流
- InterviewerAgent: 面试官，主导对话，快速响应
- ObserverAgent: 观察员，异步旁听分析，不打断

Day 21 实现：真正的异步 Agent 通信，并行处理
"""

import json
import asyncio
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ============ 数据模型 ============

class InterviewPhase(Enum):
    """面试阶段"""
    GREETING = "开场问候"
    TECHNICAL = "技术问答"
    BEHAVIORAL = "行为面试"
    SUMMARY = "总结反馈"


@dataclass
class Observation:
    """观察报告条目"""
    timestamp: str
    question_index: int
    fluency_score: int  # 1-10
    confidence_score: int  # 1-10
    clarity_score: int  # 1-10
    stutter_indicators: List[str]
    emotional_state: str
    suggestions: List[str]
    analysis_text: str = ""  # 详细分析文本


@dataclass
class InterviewContext:
    """面试上下文"""
    session_id: str
    candidate_name: str
    position: str
    current_phase: InterviewPhase = InterviewPhase.GREETING
    question_count: int = 0
    max_questions: int = 8
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    observer_analyzing: bool = False  # 观察员是否正在分析
    pending_observation: Optional[Observation] = None  # 待处理的观察结果


# ============ LangGraph 状态定义 ============

class InterviewState(TypedDict):
    """面试状态 - LangGraph 流转的核心数据结构"""
    # 基础信息
    session_id: str
    candidate_name: str
    position: str
    
    # 对话状态
    current_question: str
    last_answer: str
    question_index: int
    conversation_history: List[Dict[str, str]]
    
    # Observer 异步分析状态
    observer_status: str  # "idle" | "analyzing" | "completed"
    pending_observation: Optional[Dict[str, Any]]  # 待返回的观察结果
    observations_history: List[Dict[str, Any]]  # 历史观察记录
    
    # 面试控制
    should_end: bool
    final_report: Optional[Dict[str, Any]]
    
    # 输出
    response_to_user: Dict[str, Any]  # 返回给前端的数据


# ============ LLM 调用工具 ============

def call_llm(prompt: str, system_prompt: str, temperature: float = 0.7) -> str:
    """调用 LLM API"""
    try:
        from openai import OpenAI
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("DOUBAO_API_KEY", "")
        model = os.getenv("DOUBAO_MODEL_ID", "ep-m-20260113142855-wqkg9")
        
        if not api_key:
            # Mock 模式
            return mock_llm_response(prompt)
        
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
            temperature=temperature,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ LLM调用失败: {e}")
        return mock_llm_response(prompt)


def mock_llm_response(prompt: str) -> str:
    """Mock LLM 响应（LLM调用失败时的默认回复）"""
    if "生成面试问题" in prompt or "question" in prompt.lower():
        return "你好，欢迎参加此次面试。请先简单介绍一下你自己。"
    elif "分析" in prompt:
        return json.dumps({
            "fluency_score": 7,
            "confidence_score": 6,
            "clarity_score": 8,
            "stutter_indicators": ["轻微填充词"],
            "emotional_state": "轻度紧张",
            "suggestions": ["语速可以再慢一点"],
            "analysis": "整体表达较为流畅，逻辑清晰。"
        })
    return "你好，我是AI面试官，欢迎参加此次面试。让我们开始吧。"


# ============ Agent 实现 ============

class InterviewerAgent:
    """面试官 Agent - 主导对话，快速响应"""
    
    def __init__(self):
        self.system_prompt = """你是资深技术面试官，负责主导面试流程。

职责：
1. 根据候选人背景提出合适的面试问题
2. 保持面试节奏，8个问题后自然结束
3. 问题类型：技术能力、项目经验、解决问题能力、团队协作

风格：
- 专业、友好
- 问题循序渐进
- 不直接评价候选人的表达（这是Observer的工作）"""

    def generate_question(self, state: InterviewState) -> str:
        """生成下一个面试问题"""
        history = state.get("conversation_history", [])
        index = state.get("question_index", 0)
        
        # 构建上下文
        context = ""
        if history:
            recent = history[-4:]  # 最近2轮对话
            for h in recent:
                context += f"\n{h['role']}: {h['content'][:100]}"
        
        prompt = f"""请生成第{index + 1}个面试问题。

候选人信息：
- 姓名：{state['candidate_name']}
- 职位：{state['position']}

最近对话：{context if context else "（开场）"}

要求：
1. 只返回问题本身，不要其他内容
2. 问题要有针对性，基于之前的回答深入追问
3. 如果是第一个问题，请从自我介绍或项目经验开始"""

        result = call_llm(prompt, self.system_prompt, temperature=0.7)
        return result.strip().replace("面试官：", "").replace("问：", "")

    def should_end_interview(self, state: InterviewState) -> bool:
        """判断是否应该结束面试"""
        return state.get("question_index", 0) >= 7  # 8个问题后结束


class ObserverAgent:
    """观察员 Agent - 异步旁听分析"""
    
    def __init__(self):
        self.system_prompt = """你是专业的口吃矫正观察员，负责分析候选人的表达质量。

分析维度（1-10分）：
1. 流畅度：是否有卡顿、重复、拖音
2. 自信度：语气是否坚定，用词是否肯定  
3. 清晰度：逻辑是否清晰，重点是否突出

口吃信号检测：
- 重复词："我我我"
- 拖音："我——"
- 填充词："嗯、啊、那个"
- 语速过快（紧张信号）

情绪状态：放松/紧张/焦虑/自信/混乱

注意：你只是旁听者，不要参与对话。"""

    async def analyze_response_async(self, state: InterviewState) -> Dict[str, Any]:
        """异步分析候选人的回答（后台运行，不阻塞主流程）"""
        question = state.get("current_question", "")
        answer = state.get("last_answer", "")
        index = state.get("question_index", 0)
        
        # 先进行客观的文本分析
        stutter_indicators = []
        
        # 检测重复词（如：我我我、那个那个）
        import re
        repeat_pattern = r'(\S)\1{1,}'
        repeats = re.findall(repeat_pattern, answer)
        if repeats:
            for r in repeats[:3]:  # 最多3个
                stutter_indicators.append(f"重复词：{r}{r}")
        
        # 检测填充词
        fillers = ['嗯', '啊', '那个', '然后', '就是', '这个', '呃']
        filler_count = sum(answer.count(f) for f in fillers)
        if filler_count > 0:
            stutter_indicators.append(f"填充词：{filler_count}次")
        
        # 检测拖音（简单的破折号检测）
        if '——' in answer or '～' in answer:
            stutter_indicators.append("存在拖音")
        
        # 如果回答很短（少于10个字）
        if len(answer) < 10:
            stutter_indicators.append("回答过于简短")
        
        # 根据客观分析生成合理的评分
        base_fluency = 7
        if len(stutter_indicators) == 0:
            base_fluency = 8
        elif "重复词" in str(stutter_indicators):
            base_fluency = 5
        elif "填充词" in str(stutter_indicators):
            base_fluency = 6
        
        # 构建分析总结
        if len(stutter_indicators) == 0:
            summary = "表达流畅，无明显问题"
            analysis = f"回答'{answer}'表达简洁清晰，未发现重复词、填充词或拖音现象。"
            advice = "继续保持，可以尝试用更完整的句子表达。"
        elif "回答过于简短" in stutter_indicators and len(stutter_indicators) == 1:
            summary = "回答简短，建议展开"
            analysis = f"回答'{answer}'较为简短，建议针对问题进行更详细的阐述，补充具体细节和例子。"
            advice = "1. 回答前先思考2-3个要点 2. 用STAR法则组织语言：背景-任务-行动-结果 3. 每个观点用1-2句话展开"
        else:
            summary = f"发现{len(stutter_indicators)}个表达问题" if len(stutter_indicators) <= 2 else "存在较多表达问题"
            analysis = f"回答'{answer}'中发现了以下问题：" + "；".join(stutter_indicators) + "。建议注意语言表达的规范性。"
            advice = "1. 回答前先深呼吸，组织好语言 2. 用停顿代替填充词 3. 慢速清晰地表达"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "question_index": index,
            "fluency": base_fluency,
            "confidence": 6 if len(answer) < 10 else 7,
            "clarity": 7 if len(answer) >= 10 else 5,
            "stutter_indicators": stutter_indicators,
            "emotion": "正常" if len(stutter_indicators) == 0 else "轻度紧张",
            "analysis_summary": summary,
            "stutter_analysis": analysis,
            "improvement_advice": advice
        }

        # 模拟异步调用（实际 LLM 调用不是异步的，这里用线程池包装）
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(call_llm, prompt, self.system_prompt, 0.3)
            result = await asyncio.wrap_future(future)
        
        # 解析JSON
        try:
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            
            data = json.loads(result.strip())
            
            return {
                "timestamp": datetime.now().isoformat(),
                "question_index": index,
                "fluency": data.get("fluency_score", 5),
                "confidence": data.get("confidence_score", 5),
                "clarity": data.get("clarity_score", 5),
                "stutter_indicators": data.get("stutter_indicators", []),
                "emotion": data.get("emotional_state", "未知"),
                "analysis_summary": data.get("analysis_summary", ""),
                "stutter_analysis": data.get("stutter_analysis", ""),
                "improvement_advice": data.get("improvement_advice", "")
            }
        except Exception as e:
            print(f"⚠️ 解析观察结果失败: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "question_index": index,
                "fluency": 5,
                "confidence": 5,
                "clarity": 5,
                "stutter_indicators": [],
                "emotion": "未知",
                "suggestions": [],
                "analysis": "分析失败"
            }


# ============ LangGraph 节点函数 ============

interviewer = InterviewerAgent()
observer = ObserverAgent()


def start_interview_node(state: InterviewState) -> InterviewState:
    """开始面试节点 - 生成第一个问题"""
    print(f"\n🎯 [StartInterview] 开始面试: {state['candidate_name']}")
    
    # 生成第一个问题
    question = interviewer.generate_question(state)
    
    return {
        **state,
        "current_question": question,
        "question_index": 0,
        "conversation_history": [{"role": "interviewer", "content": question}],
        "observer_status": "idle",
        "pending_observation": None,
        "observations_history": [],
        "should_end": False,
        "response_to_user": {
            "session_id": state["session_id"],
            "status": "ongoing",
            "question": question,
            "question_index": 0,
            "observer_status": "观察员已就绪"
        }
    }


def receive_answer_node(state: InterviewState) -> InterviewState:
    """接收用户回答节点 - 立即返回，触发后台分析"""
    answer = state.get("last_answer", "")
    
    print(f"\n📝 [ReceiveAnswer] 收到回答: {answer[:50]}...")
    
    # 更新对话历史
    history = state.get("conversation_history", [])
    history.append({"role": "candidate", "content": answer})
    
    return {
        **state,
        "conversation_history": history,
        "observer_status": "analyzing",  # 标记观察员开始分析
        "response_to_user": {
            "status": "analyzing",
            "message": "观察员正在分析...",
            "question_index": state["question_index"]
        }
    }


async def observer_analyze_node(state: InterviewState) -> InterviewState:
    """Observer 分析节点 - 异步执行，不打断主流程"""
    print(f"\n👁️ [ObserverAnalyze] 开始异步分析...")
    
    # 异步调用观察员分析（后台运行）
    observation = await observer.analyze_response_async(state)
    
    # 保存观察结果到待处理区
    history = state.get("observations_history", [])
    history.append(observation)
    
    print(f"✅ [ObserverAnalyze] 分析完成: 流利度{observation['fluency']}/10")
    
    return {
        **state,
        "pending_observation": observation,
        "observations_history": history,
        "observer_status": "completed"
    }


def interviewer_respond_node(state: InterviewState) -> InterviewState:
    """面试官响应节点 - 生成下一个问题"""
    print(f"\n🎤 [InterviewerRespond] 生成下一个问题...")
    
    current_index = state.get("question_index", 0)
    new_index = current_index + 1
    
    # 检查是否结束
    if interviewer.should_end_interview({**state, "question_index": new_index}):
        return {
            **state,
            "should_end": True,
            "question_index": new_index,
            "response_to_user": {
                "session_id": state["session_id"],
                "status": "completed",
                "message": "面试结束",
                "question_index": new_index,
                "current_observation": state.get("pending_observation")
            }
        }
    
    # 生成下一个问题
    question = interviewer.generate_question({**state, "question_index": new_index})
    
    # 更新对话历史
    history = state.get("conversation_history", [])
    history.append({"role": "interviewer", "content": question})
    
    return {
        **state,
        "current_question": question,
        "question_index": new_index,
        "conversation_history": history,
        "response_to_user": {
            "session_id": state["session_id"],
            "status": "ongoing",
            "question": question,
            "question_index": new_index,
            "current_observation": state.get("pending_observation")  # 返回观察结果
        }
    }


def generate_report_node(state: InterviewState) -> InterviewState:
    """生成最终报告节点"""
    print(f"\n📊 [GenerateReport] 生成观察员报告...")
    
    observations = state.get("observations_history", [])
    
    if not observations:
        report = {
            "overall_fluency": 5,
            "overall_confidence": 5,
            "overall_clarity": 5,
            "trend": "样本不足",
            "key_issues": ["对话轮数太少，无法分析"],
            "improvement_suggestions": ["建议多练习面试场景"],
            "detailed_observations": []
        }
    else:
        # 计算平均分
        avg_fluency = sum(o["fluency"] for o in observations) / len(observations)
        avg_confidence = sum(o["confidence"] for o in observations) / len(observations)
        avg_clarity = sum(o["clarity"] for o in observations) / len(observations)
        
        # 判断趋势
        if len(observations) >= 2:
            first_fluency = observations[0]["fluency"]
            last_fluency = observations[-1]["fluency"]
            if last_fluency > first_fluency + 1:
                trend = "渐入佳境"
            elif last_fluency < first_fluency - 1:
                trend = "后期紧张"
            else:
                trend = "表现稳定"
        else:
            trend = "样本不足"
        
        # 收集所有问题
        all_issues = []
        for obs in observations:
            all_issues.extend(obs.get("stutter_indicators", []))
        
        # 收集建议
        all_suggestions = []
        for obs in observations:
            all_suggestions.extend(obs.get("suggestions", []))
        
        report = {
            "overall_fluency": round(avg_fluency, 1),
            "overall_confidence": round(avg_confidence, 1),
            "overall_clarity": round(avg_clarity, 1),
            "trend": trend,
            "key_issues": list(set(all_issues))[:3] if all_issues else ["无明显问题"],
            "improvement_suggestions": list(set(all_suggestions))[:5] if all_suggestions else ["继续保持"],
            "detailed_observations": observations
        }
    
    return {
        **state,
        "final_report": report,
        "response_to_user": {
            "session_id": state["session_id"],
            "status": "completed",
            "observer_report": report,
            "total_questions": state["question_index"]
        }
    }


# ============ 条件路由函数 ============

def should_end_or_continue(state: InterviewState) -> str:
    """判断是结束面试还是继续"""
    if state.get("should_end", False):
        return "generate_report"
    return "interviewer_respond"


def check_observer_status(state: InterviewState) -> str:
    """检查观察员分析状态"""
    status = state.get("observer_status", "idle")
    if status == "completed":
        return "analysis_done"
    return "wait_for_analysis"


# ============ 构建 LangGraph ============

def create_interview_graph() -> StateGraph:
    """创建面试工作流图"""
    
    # 定义状态图
    workflow = StateGraph(InterviewState)
    
    # 添加节点
    workflow.add_node("start_interview", start_interview_node)
    workflow.add_node("receive_answer", receive_answer_node)
    workflow.add_node("observer_analyze", observer_analyze_node)
    workflow.add_node("interviewer_respond", interviewer_respond_node)
    workflow.add_node("generate_report", generate_report_node)
    
    # 定义边
    # 1. 开始 -> 接收回答
    workflow.set_entry_point("start_interview")
    
    # 2. 接收回答后，并行触发观察员分析和等待用户（但这里我们走顺序，因为需要返回给用户）
    workflow.add_edge("start_interview", "receive_answer")
    
    # 3. 接收回答后，触发观察员分析
    workflow.add_edge("receive_answer", "observer_analyze")
    
    # 4. 观察员分析完成后，检查是否结束或继续
    workflow.add_conditional_edges(
        "observer_analyze",
        should_end_or_continue,
        {
            "generate_report": "generate_report",
            "interviewer_respond": "interviewer_respond"
        }
    )
    
    # 5. 面试官响应后，回到接收回答（循环）
    workflow.add_edge("interviewer_respond", "receive_answer")
    
    # 6. 生成报告后结束
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# ============ 异步面试管理器 ============

class AsyncInterviewManager:
    """异步面试管理器 - 支持 LangGraph 工作流"""
    
    def __init__(self):
        self.graph = create_interview_graph()
        self.memory = MemorySaver()
        self.sessions: Dict[str, InterviewState] = {}
    
    async def start_interview(self, session_id: str, candidate_name: str, position: str) -> Dict[str, Any]:
        """开始面试 - 异步"""
        print(f"\n🚀 [AsyncManager] 启动面试: {session_id}")
        
        # 初始化状态
        initial_state: InterviewState = {
            "session_id": session_id,
            "candidate_name": candidate_name,
            "position": position,
            "current_question": "",
            "last_answer": "",
            "question_index": 0,
            "conversation_history": [],
            "observer_status": "idle",
            "pending_observation": None,
            "observations_history": [],
            "should_end": False,
            "final_report": None,
            "response_to_user": {}
        }
        
        # 执行开始节点
        result = start_interview_node(initial_state)
        self.sessions[session_id] = result
        
        return result["response_to_user"]
    
    async def submit_answer(self, session_id: str, answer: str) -> Dict[str, Any]:
        """提交回答 - 异步并行处理"""
        print(f"\n📝 [AsyncManager] 提交回答: {session_id}")
        
        if session_id not in self.sessions:
            return {"error": "面试会话不存在"}
        
        # 获取当前状态
        state = self.sessions[session_id]
        state["last_answer"] = answer
        
        # 步骤1: 接收回答
        state = receive_answer_node(state)
        
        # 步骤2: 立即生成下一个问题返回给前端（不等待观察员）
        if should_end_or_continue(state) == "generate_report":
            state["should_end"] = True
            state = generate_report_node(state)
        else:
            # 先生成下一个问题立即返回
            state = interviewer_respond_node(state)
        
        # 保存状态（包含面试官的问题）
        self.sessions[session_id] = state
        response = state["response_to_user"]
        
        # 步骤3: 在后台异步执行观察员分析（不阻塞响应）
        # 创建后台任务，观察员分析完成后更新状态
        async def background_observer_analysis():
            try:
                print(f"\n👁 [Background] 观察员开始后台分析...")
                updated_state = await observer_analyze_node(self.sessions[session_id])
                self.sessions[session_id] = updated_state
                print(f"\n✅ [Background] 观察员分析完成")
            except Exception as e:
                print(f"\n⚠️ [Background] 观察员分析失败: {e}")
        
        # 启动后台任务，不等待完成
        import asyncio
        asyncio.create_task(background_observer_analysis())
        
        return response
    
    async def end_interview(self, session_id: str) -> Dict[str, Any]:
        """结束面试"""
        if session_id not in self.sessions:
            return {"error": "面试会话不存在"}
        
        state = self.sessions[session_id]
        state["should_end"] = True
        
        result = generate_report_node(state)
        self.sessions[session_id] = result
        
        return result["response_to_user"]
    
    def get_session_state(self, session_id: str) -> Optional[InterviewState]:
        """获取会话状态"""
        return self.sessions.get(session_id)


# ============ 同步包装器（兼容现有 API） ============

class InterviewManagerSync:
    """同步包装器 - 兼容现有 API 接口"""
    
    def __init__(self):
        self.async_manager = AsyncInterviewManager()
    
    def start_interview(self, session_id: str, candidate_name: str, position: str) -> Dict[str, Any]:
        """同步包装的开始面试"""
        return asyncio.run(self.async_manager.start_interview(session_id, candidate_name, position))
    
    def process_answer(self, session_id: str, answer: str) -> Dict[str, Any]:
        """同步包装的处理回答"""
        return asyncio.run(self.async_manager.submit_answer(session_id, answer))
    
    def end_interview(self, session_id: str) -> Dict[str, Any]:
        """同步包装的结束面试"""
        return asyncio.run(self.async_manager.end_interview(session_id))


# ============ 导出 ============

# 导出异步接口（给 FastAPI 使用）
async_interview_manager = AsyncInterviewManager()

# 导出给 main.py 使用的同步接口
interview_manager = InterviewManagerSync()

if __name__ == "__main__":
    # 测试
    async def test():
        manager = AsyncInterviewManager()
        
        # 开始面试
        result = await manager.start_interview("test-001", "张三", "软件工程师")
        print(f"\n开始面试: {result}")
        
        # 模拟回答
        result = await manager.submit_answer("test-001", "我之前做过一个电商项目")
        print(f"\n提交回答: {result}")
        
        # 再回答一轮
        result = await manager.submit_answer("test-001", "我主要负责后端开发")
        print(f"\n提交回答: {result}")
    
    asyncio.run(test())
