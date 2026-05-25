"""
专家团队工作流 - 基于 LangGraph
支持条件路由，根据问题类型和复杂度调用不同专家组合
"""

import json
import os
from typing import TypedDict, Literal, Optional, List, Dict, Any
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ============ 状态定义 ============

class ExpertTeamState(TypedDict):
    """专家团队工作流状态"""
    # 基础信息
    thread_id: str
    user_id: str
    user_message: str
    user_profile: Dict[str, Any]
    
    # 路由决策
    question_type: Literal["diagnosis", "data_analysis", "training_plan", "psychological", "general"]
    complexity: Literal["simple", "medium", "complex"]
    routing_reason: str
    
    # 各专家输出
    diagnosis_result: Optional[str]
    data_analysis_result: Optional[str]
    plan_result: Optional[str]
    support_result: Optional[str]
    
    # 执行记录
    experts_involved: List[str]
    execution_path: List[str]  # 记录走了哪些节点
    
    # 最终输出
    final_response: str
    status: Literal["running", "completed", "error"]


# ============ LLM 调用 ============

def call_doubao(prompt: str, system_prompt: str = "你是一个专业的口吃矫正专家。", expect_json: bool = False) -> str:
    """调用豆包 API（使用 OpenAI 兼容接口）"""
    from openai import OpenAI
    from dotenv import load_dotenv
    
    # 加载环境变量
    load_dotenv()
    
    model = os.getenv("DOUBAO_MODEL_ID", "ep-m-20260113142855-wqkg9")
    api_key = os.getenv("DOUBAO_API_KEY", "")
    
    # 如果没有 API Key，使用 Mock 模式（用于测试）
    if not api_key:
        print("⚠️  未配置 ARK_API_KEY，使用 Mock 模式")
        if expect_json:
            # 根据 prompt 内容智能判断
            if "心理" in prompt or "沮丧" in prompt or "紧张" in prompt:
                return {"question_type": "psychological", "complexity": "medium", "reason": "用户表达情绪困扰"}
            elif "数据" in prompt or "分析" in prompt or "练得怎么样" in prompt:
                return {"question_type": "data_analysis", "complexity": "medium", "reason": "用户询问训练数据"}
            elif "计划" in prompt or "制定" in prompt:
                return {"question_type": "training_plan", "complexity": "medium", "reason": "用户需要训练计划"}
            elif "卡" in prompt or "怎么回事" in prompt or "为什么" in prompt:
                return {"question_type": "diagnosis", "complexity": "medium", "reason": "用户描述症状需要诊断"}
            elif len(prompt) < 50:
                return {"question_type": "general", "complexity": "simple", "reason": "简单问候或短问题"}
            else:
                return {"question_type": "diagnosis", "complexity": "medium", "reason": "默认诊断类型"}
        return "【Mock回复】这是模拟的 AI 回复，用于测试工作流。实际使用请配置 ARK_API_KEY 环境变量。"
    
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
        # 提取 JSON
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except:
            return {
                "question_type": "general",
                "complexity": "simple",
                "reason": "解析失败，使用默认值"
            }
    
    return content


# ============ 专家节点 ============

def router_expert(state: ExpertTeamState) -> Dict:
    """
    路由专家：分析问题类型和复杂度，决定调用哪些专家
    """
    prompt = f"""你是一个问题分类专家。请分析用户问题，输出JSON格式：

用户问题：{state['user_message']}
用户画像：{json.dumps(state['user_profile'], ensure_ascii=False)}

请判断：
1. question_type: 问题类型
   - "diagnosis": 诊断类（如"我说话卡壳怎么回事"）
   - "data_analysis": 数据分析类（如"我最近练得怎么样"）
   - "training_plan": 训练计划类（如"给我制定个计划"）
   - "psychological": 心理支持类（如"我很沮丧/紧张"）
   - "general": 一般问题（如"口吃是什么"）

2. complexity: 复杂度
   - "simple": 简单问题，1个专家或直接回答
   - "medium": 中等，2-3个专家协作
   - "complex": 复杂，需要多专家深度分析

输出格式（严格JSON）：
{{
    "question_type": "diagnosis",
    "complexity": "medium", 
    "reason": "用户描述症状，需要诊断+建议"
}}"""
    
    result = call_doubao(prompt, expect_json=True)
    
    return {
        "question_type": result.get("question_type", "general"),
        "complexity": result.get("complexity", "simple"),
        "routing_reason": result.get("reason", ""),
        "execution_path": ["router_expert"]
    }


def diagnosis_expert(state: ExpertTeamState) -> Dict:
    """诊断专家：分析问题原因"""
    prompt = f"""你是口吃矫正诊断专家。请分析用户问题并给出专业诊断。

用户描述：{state['user_message']}
用户画像：{json.dumps(state['user_profile'], ensure_ascii=False)}

请给出：
1. 问题诊断（是什么类型的问题）
2. 严重程度评估  
3. 可能的原因分析
4. 建议关注的重点

回答要专业、温暖，让用户感到被理解。"""
    
    result = call_doubao(prompt)
    
    return {
        "diagnosis_result": result,
        "execution_path": state.get("execution_path", []) + ["diagnosis_expert"]
    }


def data_analyst(state: ExpertTeamState) -> Dict:
    """数据分析师：分析训练数据"""
    # 获取训练数据（这里简化，实际需要查数据库）
    training_data = state.get('user_profile', {}).get('training_summary', '暂无训练数据')
    
    prompt = f"""你是数据分析师，负责分析用户的训练数据。

用户近期训练情况：{training_data}
用户问题：{state['user_message']}

请分析：
1. 训练数据反映的问题或进步
2. 与当前问题的关联
3. 数据支撑的建议

如果数据不足，请说明需要补充哪些信息。"""
    
    result = call_doubao(prompt)
    
    return {
        "data_analysis_result": result,
        "execution_path": state.get("execution_path", []) + ["data_analyst"]
    }


def plan_expert(state: ExpertTeamState) -> Dict:
    """方案专家：制定训练建议"""
    # 整合之前的分析结果
    context = []
    if state.get('diagnosis_result'):
        context.append(f"诊断结果：{state['diagnosis_result'][:200]}...")
    if state.get('data_analysis_result'):
        context.append(f"数据分析：{state['data_analysis_result'][:200]}...")
    
    context_str = "\n".join(context) if context else "暂无前置分析"
    
    # 获取用户画像信息
    user_profile = state.get('user_profile', {})
    severity_level = user_profile.get('severity_level', 3)
    stuttering_type = user_profile.get('stuttering_type', '未知')
    strengths = user_profile.get('strengths', [])
    weaknesses = user_profile.get('weaknesses', [])
    effective_methods = user_profile.get('effective_methods', [])
    recommended_focus = user_profile.get('recommended_focus', '')
    weekly_progress = user_profile.get('weekly_progress', 0)
    trend_direction = user_profile.get('trend_direction', 'stable')
    
    # 构建优势/弱项描述
    strengths_str = '、'.join(strengths) if strengths else '暂无记录'
    weaknesses_str = '、'.join(weaknesses) if weaknesses else '暂无记录'
    methods_str = '、'.join(effective_methods) if effective_methods else '暂无记录'
    
    prompt = f"""你是训练方案专家，请基于用户画像和已有分析给出个性化建议。

用户画像：
- 严重程度：{severity_level}/5级
- 口吃类型：{stuttering_type}
- 优势：{strengths_str}
- 弱项：{weaknesses_str}
- 有效方法：{methods_str}
- 建议重点：{recommended_focus or '暂无'}
- 周进步：{weekly_progress:.1f}% ({trend_direction})

用户问题：{state['user_message']}

前置分析：
{context_str}

请给出：
1. 具体的训练建议（可执行）
2. 推荐的训练方法/技巧（结合用户等级和类型）
3. 注意事项（针对用户特点）
4. 预期效果

建议要具体、可操作，避免空泛。"""
    
    result = call_doubao(prompt)
    
    return {
        "plan_result": result,
        "execution_path": state.get("execution_path", []) + ["plan_expert"]
    }


def support_expert(state: ExpertTeamState, intensive: bool = False) -> Dict:
    """
    心理支持专家
    intensive=True 时提供更深入的心理支持
    """
    depth = "深入" if intensive else "一般"
    
    # 获取用户画像信息
    user_profile = state.get('user_profile', {})
    severity_level = user_profile.get('severity_level', 3)
    stuttering_type = user_profile.get('stuttering_type', '未知')
    strengths = user_profile.get('strengths', [])
    weekly_progress = user_profile.get('weekly_progress', 0)
    trend_direction = user_profile.get('trend_direction', 'stable')
    
    # 构建用户背景描述
    level_desc = {1: '轻度', 2: '轻度', 3: '中度', 4: '较重', 5: '重度'}.get(severity_level, '未知')
    background = f"该用户是{level_desc}水平（{severity_level}/5级）"
    if stuttering_type != '未知':
        background += f"，{stuttering_type}"
    
    # 根据进步情况调整鼓励策略
    progress_desc = ""
    if weekly_progress > 10:
        progress_desc = "本周进步明显，"
    elif weekly_progress < -10:
        progress_desc = "本周遇到瓶颈，"
    
    prompt = f"""你是心理支持专家，提供{depth}心理支持。

用户背景：{background}
用户优势：{'、'.join(strengths) if strengths else '暂无记录'}
近期进展：{progress_desc}周进步{weekly_progress:.1f}%

用户表达：{state['user_message']}

请给出：
1. 共情和认可（让用户感到被理解，结合其严重程度{level_desc}）
2. 鼓励话语（针对{level_desc}水平用户的具体鼓励）
3. 心态调整建议（结合其优势和近期{trend_direction}趋势给出）
{"4. 深度心理疏导（针对焦虑/抑郁情绪）" if intensive else ""}

语气要温暖、支持，避免说教。"""
    
    result = call_doubao(prompt)
    
    return {
        "support_result": result,
        "execution_path": state.get("execution_path", []) + [f"support_expert_{'intensive' if intensive else 'normal'}"]
    }


def aggregator(state: ExpertTeamState) -> Dict:
    """汇总专家：整合所有结果生成最终回复"""
    
    # 收集参与的专家结果
    parts = []
    experts = []
    
    if state.get('diagnosis_result'):
        parts.append(f"【诊断分析】\n{state['diagnosis_result']}")
        experts.append("诊断专家")
    
    if state.get('data_analysis_result'):
        parts.append(f"【数据分析】\n{state['data_analysis_result']}")
        experts.append("数据分析师")
    
    if state.get('plan_result'):
        parts.append(f"【训练建议】\n{state['plan_result']}")
        experts.append("方案专家")
    
    if state.get('support_result'):
        parts.append(f"【心理支持】\n{state['support_result']}")
        experts.append("心理支持专家")
    
    # 获取用户画像
    user_profile = state.get('user_profile', {})
    severity_level = user_profile.get('severity_level', 3)
    stuttering_type = user_profile.get('stuttering_type', '')
    recommended_focus = user_profile.get('recommended_focus', '')
    
    # 构建画像描述
    level_desc = {1: '轻度', 2: '轻度', 3: '中度', 4: '较重', 5: '重度'}.get(severity_level, '未知')
    profile_desc = f"{level_desc}水平（{severity_level}/5级）"
    if stuttering_type:
        profile_desc += f"，{stuttering_type}"
    
    # 如果是简单问题，没有专家输出，直接回复
    if not parts:
        prompt = f"""请直接回复用户的问题：

用户画像：{profile_desc}
{f"AI建议重点：{recommended_focus}" if recommended_focus else ""}
用户问题：{state['user_message']}

要求：简洁、温暖、专业，考虑用户{level_desc}水平和特点。"""
        final = call_doubao(prompt)
        experts = ["通用回复"]
    else:
        # 构建内容部分（避免 f-string 中使用复杂表达式）
        separator = "=" * 50
        combined_parts = "\n\n".join(parts)
        
        prompt = f"""你是汇总专家，请综合以下内容生成最终回复。

用户画像：{profile_desc}
{f"AI建议重点：{recommended_focus}" if recommended_focus else ""}

用户问题：{state['user_message']}

{separator}
{combined_parts}
{separator}

要求：
1. 开头要有温度（共情），考虑用户当前{level_desc}水平阶段
2. 整合各专家观点，不要简单拼接
3. 结构清晰，用emoji增加可读性
4. 给出可执行的下步行动（适合{level_desc}水平）
5. 总字数控制在400字以内"""
        
        final = call_doubao(prompt)
    
    return {
        "final_response": final,
        "experts_involved": experts,
        "execution_path": state.get("execution_path", []) + ["aggregator"],
        "status": "completed"
    }


# ============ 条件路由函数 ============

def route_by_type(state: ExpertTeamState) -> str:
    """
    根据问题类型和复杂度路由到不同节点
    """
    qtype = state.get('question_type', 'general')
    complexity = state.get('complexity', 'simple')
    
    # 简单问题：直接汇总（快速回复）
    if complexity == 'simple':
        return 'aggregator'
    
    # 心理问题：直接深度支持
    if qtype == 'psychological':
        return 'support_expert_intensive'
    
    # 数据分析类：直接数据分析
    if qtype == 'data_analysis':
        return 'data_analyst'
    
    # 训练计划类：直接方案专家
    if qtype == 'training_plan':
        return 'plan_expert'
    
    # 诊断类/一般问题：从诊断开始
    return 'diagnosis_expert'


def route_after_diagnosis(state: ExpertTeamState) -> str:
    """诊断后路由"""
    qtype = state.get('question_type')
    complexity = state.get('complexity')
    
    # 复杂问题需要数据分析
    if complexity == 'complex':
        return 'data_analyst'
    
    # 诊断类问题继续走方案
    if qtype == 'diagnosis':
        return 'plan_expert'
    
    # 其他直接汇总
    return 'aggregator'


def route_after_data(state: ExpertTeamState) -> str:
    """数据分析后路由"""
    # 数据分析后都需要方案
    return 'plan_expert'


def route_after_plan(state: ExpertTeamState) -> str:
    """方案后路由"""
    qtype = state.get('question_type')
    complexity = state.get('complexity')
    
    # 心理问题或复杂问题需要支持
    if qtype == 'psychological' or complexity == 'complex':
        return 'support_expert_normal'
    
    return 'aggregator'


def route_after_support(state: ExpertTeamState) -> str:
    """心理支持后路由"""
    return 'aggregator'


# ============ 构建工作流 ============

def create_expert_team_workflow():
    """
    创建带条件判断的专家团队工作流
    
    流程图：
    
    START → router_expert (智能分类)
              │
              ├─ simple ────────────────────────→ aggregator (快速回复)
              │
              ├─ psychological ─→ support_expert_intensive ─┐
              │                                               │
              ├─ data_analysis ─→ data_analyst ──────────────┤
              │                                               │
              ├─ training_plan ─→ plan_expert ────────────────┤
              │                                               │
              └─ diagnosis ─────→ diagnosis_expert ─→ plan_expert
                                                         │
              ┌──────────────────────────────────────────────┤
              │                                              │
         support_expert (可选) ←─── complex问题              │
              │                                              │
              └──────────────────→ aggregator ←─────────────┘
                                    │
                                   END
    """
    workflow = StateGraph(ExpertTeamState)
    
    # 添加节点
    workflow.add_node("router_expert", router_expert)
    workflow.add_node("diagnosis_expert", diagnosis_expert)
    workflow.add_node("data_analyst", data_analyst)
    workflow.add_node("plan_expert", plan_expert)
    workflow.add_node("support_expert_normal", lambda s: support_expert(s, intensive=False))
    workflow.add_node("support_expert_intensive", lambda s: support_expert(s, intensive=True))
    workflow.add_node("aggregator", aggregator)
    
    # 入口
    workflow.set_entry_point("router_expert")
    
    # 条件分支1：从路由到不同专家
    workflow.add_conditional_edges(
        "router_expert",
        route_by_type,
        {
            "diagnosis_expert": "diagnosis_expert",
            "data_analyst": "data_analyst",
            "plan_expert": "plan_expert",
            "support_expert_intensive": "support_expert_intensive",
            "aggregator": "aggregator",
        }
    )
    
    # 条件分支2：诊断后路由
    workflow.add_conditional_edges(
        "diagnosis_expert",
        route_after_diagnosis,
        {
            "data_analyst": "data_analyst",
            "plan_expert": "plan_expert",
            "aggregator": "aggregator",
        }
    )
    
    # 数据分析后 -> 方案专家
    workflow.add_edge("data_analyst", "plan_expert")
    
    # 条件分支3：方案后路由
    workflow.add_conditional_edges(
        "plan_expert",
        route_after_plan,
        {
            "support_expert_normal": "support_expert_normal",
            "aggregator": "aggregator",
        }
    )
    
    # 心理支持 -> 汇总
    workflow.add_edge("support_expert_intensive", "aggregator")
    workflow.add_edge("support_expert_normal", "aggregator")
    
    # 汇总 -> 结束
    workflow.add_edge("aggregator", END)
    
    # 添加内存检查点
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)


# ============ 单例 ============

_workflow = None

def get_expert_workflow():
    """获取工作流实例（单例）"""
    global _workflow
    if _workflow is None:
        _workflow = create_expert_team_workflow()
    return _workflow
