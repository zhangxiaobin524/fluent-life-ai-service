"""
面试双Agent协作系统
- InterviewerAgent: 面试官，主导对话
- ObserverAgent: 观察员，分析候选人表现
"""

import json
import asyncio
from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import os


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
    stutter_indicators: List[str]  # 检测到的口吃信号
    emotional_state: str  # 情绪状态
    suggestions: List[str]


@dataclass
class InterviewContext:
    """面试上下文"""
    candidate_name: str
    position: str
    current_phase: InterviewPhase = InterviewPhase.GREETING
    question_count: int = 0
    max_questions: int = 8
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "candidate_name": self.candidate_name,
            "position": self.position,
            "current_phase": self.current_phase.value,
            "question_count": self.question_count,
            "max_questions": self.max_questions,
            "conversation_history": self.conversation_history,
            "observations": [
                {
                    "timestamp": obs.timestamp,
                    "question_index": obs.question_index,
                    "fluency_score": obs.fluency_score,
                    "confidence_score": obs.confidence_score,
                    "clarity_score": obs.clarity_score,
                    "stutter_indicators": obs.stutter_indicators,
                    "emotional_state": obs.emotional_state,
                    "suggestions": obs.suggestions
                }
                for obs in self.observations
            ]
        }


def call_llm(prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
    """调用LLM（使用现有配置）"""
    try:
        from openai import OpenAI
        
        api_key = os.getenv("DOUBAO_API_KEY", "")
        model = os.getenv("DOUBAO_MODEL_ID", "ep-m-20260113142855-wqkg9")
        
        if not api_key:
            # Mock模式
            if "面试" in prompt or "介绍" in prompt:
                return "请简单介绍一下你自己，以及为什么应聘这个岗位？"
            elif "技术" in prompt or "项目" in prompt:
                return "请描述一下你最有挑战性的项目，遇到了什么困难？"
            return "还有什么想补充的吗？"
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature=temperature,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM调用错误: {e}")
        return "请继续..."


class InterviewerAgent:
    """
    面试官 Agent
    
    职责：
    1. 主导面试流程
    2. 根据阶段提问
    3. 评估回答质量
    4. 决定下一步
    
    限制：不能直接观察非语言信息（依赖观察员报告）
    """
    
    def __init__(self):
        self.system_prompt = """你是资深技术面试官，负责评估候选人的技术能力和沟通表达。

面试流程：
1. 开场问候（1-2题）- 让候选人放松
2. 技术问答（3-4题）- 考察专业能力
3. 行为面试（2-3题）- 考察软实力
4. 总结反馈（1题）- 收尾

提问原则：
- 问题要具体，一次只问一个点
- 根据候选人回答调整难度
- 保持专业但友好的态度
- 如果候选人紧张，给予适当鼓励"""
    
    def generate_question(self, context: InterviewContext) -> str:
        """生成下一个问题"""
        # 判断当前阶段
        if context.question_count == 0:
            context.current_phase = InterviewPhase.GREETING
        elif context.question_count <= 2:
            context.current_phase = InterviewPhase.TECHNICAL
        elif context.question_count <= 5:
            context.current_phase = InterviewPhase.BEHAVIORAL
        else:
            context.current_phase = InterviewPhase.SUMMARY
        
        # 构建提示
        history_text = "\n".join([
            f"{'面试官' if msg['role'] == 'interviewer' else '候选人'}：{msg['content'][:100]}..."
            for msg in context.conversation_history[-4:]
        ]) if context.conversation_history else "暂无对话"
        
        prompt = f"""当前面试阶段：{context.current_phase.value}
已提问数：{context.question_count}/{context.max_questions}
应聘岗位：{context.position}

近期对话：
{history_text}

请生成下一个{context.current_phase.value}阶段的问题。
要求：
1. 符合当前阶段特点
2. 问题简洁具体
3. 直接输出问题内容，不要解释"""
        
        question = call_llm(prompt, self.system_prompt)
        
        # 清理输出
        question = question.strip().strip('"').strip("'")
        if "：" in question:
            question = question.split("：", 1)[-1].strip()
        
        return question
    
    def should_end_interview(self, context: InterviewContext) -> bool:
        """判断是否结束面试"""
        return context.question_count >= context.max_questions


class ObserverAgent:
    """
    观察员 Agent
    
    职责：
    1. 旁听对话，不直接参与
    2. 分析候选人的表达质量
    3. 检测口吃信号和紧张表现
    4. 生成观察报告
    
    限制：不与候选人直接对话
    """
    
    def __init__(self):
        self.system_prompt = """你是专业的口吃矫正观察员，负责分析候选人的表达表现。

分析维度：
1. 流畅度（1-10）：是否有卡顿、重复、拖音
2. 自信度（1-10）：语气是否坚定，用词是否肯定
3. 清晰度（1-10）：逻辑是否清晰，重点是否突出

口吃信号检测：
- 重复词："我我我"
- 拖音："我——"
- 填充词："嗯、啊、那个"
- 突然停顿
- 语速过快（紧张）

情绪状态判断：
- 放松：语速适中，用词自然
- 紧张：语速快，填充词多
- 焦虑：句子破碎，重复多
- 自信：语速稳定，表达流畅"""
    
    def analyze_response(self, question: str, answer: str, question_index: int) -> Observation:
        """分析一次回答"""
        prompt = f"""请分析候选人的这次回答。

面试官问题：{question}
候选人回答：{answer}

请以JSON格式返回分析结果：
{{
    "fluency_score": 7,
    "confidence_score": 6,
    "clarity_score": 8,
    "stutter_indicators": ["填充词：嗯", "轻微拖音"],
    "emotional_state": "轻度紧张",
    "suggestions": ["可以尝试放慢语速", "减少填充词使用"]
}}

只返回JSON，不要其他内容。"""
        
        result = call_llm(prompt, self.system_prompt, temperature=0.3)
        
        # 解析JSON
        try:
            # 提取JSON部分
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            
            data = json.loads(result.strip())
            
            return Observation(
                timestamp=datetime.now().isoformat(),
                question_index=question_index,
                fluency_score=data.get("fluency_score", 5),
                confidence_score=data.get("confidence_score", 5),
                clarity_score=data.get("clarity_score", 5),
                stutter_indicators=data.get("stutter_indicators", []),
                emotional_state=data.get("emotional_state", "未知"),
                suggestions=data.get("suggestions", [])
            )
        except Exception as e:
            print(f"解析观察结果失败: {e}")
            return Observation(
                timestamp=datetime.now().isoformat(),
                question_index=question_index,
                fluency_score=5,
                confidence_score=5,
                clarity_score=5,
                stutter_indicators=["分析失败"],
                emotional_state="未知",
                suggestions=["请重试"]
            )
    
    def generate_final_report(self, context: InterviewContext) -> Dict[str, Any]:
        """生成最终观察报告"""
        if not context.observations:
            return {
                "overall_fluency": 5,
                "overall_confidence": 5,
                "overall_clarity": 5,
                "trend": "稳定",
                "key_issues": ["样本不足，无法分析"],
                "strengths": [],
                "improvement_suggestions": ["建议多练习面试场景"],
                "detailed_observations": []
            }
        
        # 计算平均分
        avg_fluency = sum(o.fluency_score for o in context.observations) / len(context.observations)
        avg_confidence = sum(o.confidence_score for o in context.observations) / len(context.observations)
        avg_clarity = sum(o.clarity_score for o in context.observations) / len(context.observations)
        
        # 计算趋势（前半场 vs 后半场）
        mid = len(context.observations) // 2
        first_half = context.observations[:mid] if mid > 0 else context.observations
        second_half = context.observations[mid:] if mid > 0 else context.observations
        
        first_avg = sum(o.fluency_score for o in first_half) / len(first_half) if first_half else 0
        second_avg = sum(o.fluency_score for o in second_half) / len(second_half) if second_half else 0
        
        if second_avg - first_avg > 1:
            trend = "渐入佳境"
        elif second_avg - first_avg < -1:
            trend = "后期紧张"
        else:
            trend = "表现稳定"
        
        # 收集所有问题
        all_issues = []
        for obs in context.observations:
            all_issues.extend(obs.stutter_indicators)
        
        # 统计频率高的问题
        from collections import Counter
        issue_counts = Counter(all_issues)
        key_issues = [issue for issue, count in issue_counts.most_common(3) if count >= 1]
        
        # 提取建议
        all_suggestions = []
        for obs in context.observations:
            all_suggestions.extend(obs.suggestions)
        unique_suggestions = list(set(all_suggestions))[:5]
        
        return {
            "overall_fluency": round(avg_fluency, 1),
            "overall_confidence": round(avg_confidence, 1),
            "overall_clarity": round(avg_clarity, 1),
            "trend": trend,
            "key_issues": key_issues if key_issues else ["无明显问题"],
            "strengths": self._identify_strengths(context.observations),
            "improvement_suggestions": unique_suggestions if unique_suggestions else ["继续保持"],
            "detailed_observations": [
                {
                    "question_index": obs.question_index,
                    "fluency": obs.fluency_score,
                    "confidence": obs.confidence_score,
                    "clarity": obs.clarity_score,
                    "emotion": obs.emotional_state
                }
                for obs in context.observations
            ]
        }
    
    def _identify_strengths(self, observations: List[Observation]) -> List[str]:
        """识别优势"""
        strengths = []
        
        avg_confidence = sum(o.confidence_score for o in observations) / len(observations)
        avg_clarity = sum(o.clarity_score for o in observations) / len(observations)
        avg_fluency = sum(o.fluency_score for o in observations) / len(observations)
        
        if avg_confidence >= 7:
            strengths.append("整体表现自信")
        if avg_clarity >= 7:
            strengths.append("表达逻辑清晰")
        if avg_fluency >= 7:
            strengths.append("语言流畅度高")
        
        # 检查进步趋势
        if len(observations) >= 2:
            first = observations[0].fluency_score
            last = observations[-1].fluency_score
            if last > first + 1:
                strengths.append("适应能力强，渐入佳境")
        
        return strengths if strengths else ["表现平稳"]


class InterviewManager:
    """
    面试管理器
    协调InterviewerAgent和ObserverAgent
    """
    
    def __init__(self):
        self.interviewer = InterviewerAgent()
        self.observer = ObserverAgent()
        self.contexts: Dict[str, InterviewContext] = {}
    
    def start_interview(self, session_id: str, candidate_name: str, position: str) -> Dict[str, Any]:
        """开始新面试"""
        context = InterviewContext(
            candidate_name=candidate_name,
            position=position
        )
        self.contexts[session_id] = context
        
        # 生成第一个问题
        question = self.interviewer.generate_question(context)
        context.conversation_history.append({
            "role": "interviewer",
            "content": question
        })
        
        return {
            "session_id": session_id,
            "question": question,
            "question_index": 0,
            "observer_status": "观察员已就绪，正在记录..."
        }
    
    def process_answer(self, session_id: str, answer: str) -> Dict[str, Any]:
        """处理候选人回答"""
        print(f"📥 [面试] 收到回答 - session_id: {session_id}, answer: {answer[:50]}...")
        
        if session_id not in self.contexts:
            print(f"❌ [面试] 会话不存在: {session_id}")
            return {"error": "面试会话不存在"}
        
        context = self.contexts[session_id]
        print(f"📊 [面试] 当前进度: {context.question_count}/{context.max_questions}")
        
        # 1. Observer分析这次回答
        last_question = context.conversation_history[-1]["content"] if context.conversation_history else ""
        print(f"📝 [面试] 分析问题 - 问题: {last_question[:50]}...")
        observation = self.observer.analyze_response(
            last_question, 
            answer, 
            context.question_count
        )
        context.observations.append(observation)
        print(f"✅ [面试] 观察分析完成 - 流畅度: {observation.fluency_score}, 情绪: {observation.emotional_state}")
        
        # 2. 记录回答
        context.conversation_history.append({
            "role": "candidate",
            "content": answer
        })
        context.question_count += 1
        
        # 3. 检查是否结束
        if self.interviewer.should_end_interview(context):
            report = self.observer.generate_final_report(context)
            return {
                "status": "completed",
                "message": "面试结束",
                "observer_report": report,
                "total_questions": context.question_count
            }
        
        # 4. Interviewer生成下一个问题
        print(f"🤖 [面试] InterviewerAgent 生成下一个问题...")
        next_question = self.interviewer.generate_question(context)
        context.conversation_history.append({
            "role": "interviewer",
            "content": next_question
        })
        print(f"✅ [面试] 生成问题完成: {next_question[:50]}...")
        
        return {
            "status": "ongoing",
            "question": next_question,
            "question_index": context.question_count,
            "current_observation": {
                "fluency": observation.fluency_score,
                "confidence": observation.confidence_score,
                "clarity": observation.clarity_score,
                "emotion": observation.emotional_state
            }
        }
    
    def get_observer_report(self, session_id: str) -> Dict[str, Any]:
        """获取观察报告"""
        if session_id not in self.contexts:
            return {"error": "面试会话不存在"}
        
        context = self.contexts[session_id]
        report = self.observer.generate_final_report(context)
        report["conversation_history"] = context.conversation_history
        
        return report
    
    def end_interview(self, session_id: str) -> Dict[str, Any]:
        """结束面试"""
        if session_id not in self.contexts:
            return {"error": "面试会话不存在"}
        
        context = self.contexts[session_id]
        report = self.observer.generate_final_report(context)
        
        # 清理会话
        del self.contexts[session_id]
        
        return {
            "status": "ended",
            "observer_report": report,
            "total_questions": context.question_count
        }


# 全局管理器实例
interview_manager = InterviewManager()
