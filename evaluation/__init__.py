"""
LLM-as-a-Judge 评测系统
用于评估 AI 模块输出质量
"""

from .evaluator import AIEvaluator, LLMJudge, EvaluationResult
from .test_cases import TEST_CASES, EVALUATION_DIMENSIONS

__all__ = ['AIEvaluator', 'LLMJudge', 'EvaluationResult', 'TEST_CASES', 'EVALUATION_DIMENSIONS']
