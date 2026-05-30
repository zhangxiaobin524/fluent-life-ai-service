"""
AI 评测系统
- LLM-as-a-Judge: 评估 AI 模块输出质量
- RAGAS: 评估 RAG 系统检索和生成质量
"""

from .evaluator import AIEvaluator, LLMJudge, EvaluationResult
from .test_cases import TEST_CASES, EVALUATION_DIMENSIONS

# RAGAS 评测（可选导入）
try:
    from .ragas_evaluator import RAGASEvaluator, RAGTestCase, RAG_TEST_CASES
    __all__ = [
        'AIEvaluator', 'LLMJudge', 'EvaluationResult',
        'TEST_CASES', 'EVALUATION_DIMENSIONS',
        'RAGASEvaluator', 'RAGTestCase', 'RAG_TEST_CASES'
    ]
except ImportError:
    __all__ = ['AIEvaluator', 'LLMJudge', 'EvaluationResult', 'TEST_CASES', 'EVALUATION_DIMENSIONS']
