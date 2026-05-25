"""
评测测试用例集
每个模块 10 个测试用例
"""

from typing import List, Dict, Any

# ========================================
# ExpertTeam 评测维度
# ========================================
EXPERT_TEAM_DIMENSIONS = {
    "路由准确性": 30,  # 正确识别问题类型，调用合适的专家
    "专家协作度": 30,  # 多专家意见整合协调
    "回答完整性": 20,  # 覆盖问题的各个层面
    "专业度": 20,      # 口吃矫正领域的专业程度
}

# ========================================
# TrainingPlan 评测维度
# ========================================
TRAINING_PLAN_DIMENSIONS = {
    "计划可行性": 30,  # 用户能够实际执行
    "个性化程度": 30,  # 符合用户个人情况
    "目标明确性": 20,  # 目标清晰可量化
    "进阶合理性": 20,  # 难度递进合理
}

# 汇总所有维度
EVALUATION_DIMENSIONS = {
    "expert_team": EXPERT_TEAM_DIMENSIONS,
    "training_plan": TRAINING_PLAN_DIMENSIONS,
}

# ========================================
# ExpertTeam 测试用例 (10个)
# ========================================
EXPERT_TEAM_TEST_CASES: List[Dict[str, Any]] = [
    {
        "id": "EXP_001",
        "name": "心理障碍咨询",
        "user_message": "我明天要面试了，现在特别紧张，说话都结巴，该怎么办？",
        "user_profile": {
            "age": 25,
            "occupation": "程序员",
            "stuttering_years": 5
        },
        "expected": {
            "question_type": "psychological",
            "should_involve": ["心理支持专家"],
            "should_not_involve": ["数据分析师"],
            "key_elements": ["紧张", "面试", "心理调节"]
        }
    },
    {
        "id": "EXP_002",
        "name": "训练数据分析",
        "user_message": "我练了一个月了，帮我看看练得怎么样？",
        "user_profile": {
            "training_days": 30,
            "total_exercises": 120
        },
        "expected": {
            "question_type": "data_analysis",
            "should_involve": ["数据分析师"],
            "key_elements": ["训练数据", "进度", "效果评估"]
        }
    },
    {
        "id": "EXP_003",
        "name": "训练计划制定",
        "user_message": "我想制定一个系统的训练计划，每天可以练习1小时",
        "user_profile": {
            "available_time": "1小时/天",
            "level": "初级"
        },
        "expected": {
            "question_type": "training_plan",
            "should_involve": ["方案专家"],
            "key_elements": ["计划", "时间安排", "系统训练"]
        }
    },
    {
        "id": "EXP_004",
        "name": "症状诊断",
        "user_message": "我说话总是第一个字重复，比如'我我我想吃饭'，这是什么类型的口吃？",
        "user_profile": {},
        "expected": {
            "question_type": "diagnosis",
            "should_involve": ["诊断专家"],
            "key_elements": ["首字重复", "类型判断", "原因分析"]
        }
    },
    {
        "id": "EXP_005",
        "name": "复杂问题（多维度）",
        "user_message": "我最近工作压力大，口吃变严重了，练了一个月也没效果，很沮丧，你能帮我分析原因并制定新计划吗？",
        "user_profile": {
            "training_days": 30,
            "stress_level": "高"
        },
        "expected": {
            "question_type": "complex",
            "should_involve": ["诊断专家", "数据分析师", "心理支持专家", "方案专家"],
            "complexity": "complex"
        }
    },
    {
        "id": "EXP_006",
        "name": "简单问候",
        "user_message": "你好",
        "user_profile": {},
        "expected": {
            "question_type": "general",
            "complexity": "simple",
            "should_involve": []  # 简单问题直接回答
        }
    },
    {
        "id": "EXP_007",
        "name": "技巧询问",
        "user_message": "气流法具体怎么做？",
        "user_profile": {
            "known_techniques": ["慢速朗读"]
        },
        "expected": {
            "question_type": "training_plan",
            "key_elements": ["技巧讲解", "步骤说明"]
        }
    },
    {
        "id": "EXP_008",
        "name": "情绪支持",
        "user_message": "我觉得自己永远也好不了了，很绝望",
        "user_profile": {
            "training_history": "多次尝试失败"
        },
        "expected": {
            "question_type": "psychological",
            "should_involve": ["心理支持专家"],
            "key_elements": ["情绪支持", "鼓励", "希望重建"]
        }
    },
    {
        "id": "EXP_009",
        "name": "效果对比",
        "user_message": "对比上个月，我这个月进步了多少？",
        "user_profile": {
            "has_historical_data": True
        },
        "expected": {
            "question_type": "data_analysis",
            "key_elements": ["对比分析", "进步量化"]
        }
    },
    {
        "id": "EXP_010",
        "name": "触发场景咨询",
        "user_message": "为什么我在打电话时特别容易口吃？",
        "user_profile": {},
        "expected": {
            "question_type": "diagnosis",
            "key_elements": ["触发因素", "场景分析", "原因解释"]
        }
    }
]

# ========================================
# TrainingPlan 测试用例 (10个)
# ========================================
TRAINING_PLAN_TEST_CASES: List[Dict[str, Any]] = [
    {
        "id": "PLAN_001",
        "name": "初级用户基础计划",
        "user_profile": {
            "stuttering_type": "首字难发",
            "severity_level": 6,
            "trigger_factors": ["紧张", "电话"],
            "recommended_techniques": ["气流法"],
            "available_time": "30分钟/天",
            "level": "初级"
        },
        "training_history": [],
        "expected": {
            "duration_days": (7, 14),
            "daily_time": (20, 40),
            "should_include": ["呼吸训练", "气流法"],
            "difficulty": "easy"
        }
    },
    {
        "id": "PLAN_002",
        "name": "中级用户进阶计划",
        "user_profile": {
            "stuttering_type": "重复发音",
            "severity_level": 4,
            "trigger_factors": ["开会", "汇报"],
            "recommended_techniques": ["慢速朗读", "轻起音"],
            "available_time": "1小时/天",
            "level": "中级"
        },
        "training_history": [
            {"date": "2024-01", "exercise": "呼吸训练", "completion_rate": 0.9}
        ],
        "expected": {
            "duration_days": (14, 21),
            "should_include": ["实战模拟", "脱敏训练"],
            "difficulty": "medium"
        }
    },
    {
        "id": "PLAN_003",
        "name": "高级用户维持计划",
        "user_profile": {
            "stuttering_type": "间歇性",
            "severity_level": 2,
            "trigger_factors": ["极度疲劳"],
            "recommended_techniques": ["维持训练"],
            "available_time": "20分钟/天",
            "level": "高级"
        },
        "training_history": [
            {"date": "2024-01", "type": "系统训练", "duration": "3个月"}
        ],
        "expected": {
            "should_include": ["维持练习", "巩固技巧"],
            "difficulty": "maintenance"
        }
    },
    {
        "id": "PLAN_004",
        "name": "时间紧张用户计划",
        "user_profile": {
            "stuttering_type": "重复发音",
            "severity_level": 5,
            "trigger_factors": ["快节奏场景"],
            "recommended_techniques": ["快速放松"],
            "available_time": "15分钟/天",
            "level": "初级"
        },
        "training_history": [],
        "expected": {
            "daily_time": (10, 20),
            "should_include": ["快速训练", "核心技巧"],
            "note": "时间虽短但要有效果"
        }
    },
    {
        "id": "PLAN_005",
        "name": "严重口吃康复计划",
        "user_profile": {
            "stuttering_type": "多重症状",
            "severity_level": 9,
            "trigger_factors": ["几乎所有场景"],
            "recommended_techniques": ["基础重建"],
            "available_time": "2小时/天",
            "level": "初级"
        },
        "training_history": [],
        "expected": {
            "duration_days": (21, 30),
            "should_include": ["基础训练", "大量重复"],
            "difficulty": "hard",
            "pace": "slow"
        }
    },
    {
        "id": "PLAN_006",
        "name": "儿童用户专项计划",
        "user_profile": {
            "stuttering_type": "发育性",
            "severity_level": 5,
            "age": 8,
            "trigger_factors": ["兴奋", "着急"],
            "recommended_techniques": ["游戏化训练"],
            "available_time": "20分钟/天",
            "level": "初级"
        },
        "training_history": [],
        "expected": {
            "should_include": ["游戏化", "家长参与", "趣味性"],
            "note": "适合儿童年龄特点"
        }
    },
    {
        "id": "PLAN_007",
        "name": "职场人士实战计划",
        "user_profile": {
            "stuttering_type": "情境性",
            "severity_level": 5,
            "occupation": "销售",
            "trigger_factors": ["客户沟通", "会议发言"],
            "recommended_techniques": ["实战演练"],
            "available_time": "45分钟/天",
            "level": "中级"
        },
        "training_history": [],
        "expected": {
            "should_include": ["场景模拟", "客户对话", "电话练习"],
            "note": "贴近工作场景"
        }
    },
    {
        "id": "PLAN_008",
        "name": "反馈调整测试",
        "user_profile": {
            "stuttering_type": "重复",
            "severity_level": 5,
            "available_time": "30分钟/天",
            "level": "初级"
        },
        "training_history": [],
        "feedback_scenario": {
            "initial": "正常生成",
            "feedback": "too_hard",
            "expected_adjustment": "降低难度"
        }
    },
    {
        "id": "PLAN_009",
        "name": "特定场景突破计划",
        "user_profile": {
            "stuttering_type": "电话恐惧",
            "severity_level": 7,
            "trigger_factors": ["电话"],
            "recommended_techniques": ["渐进暴露"],
            "available_time": "40分钟/天",
            "level": "中级"
        },
        "training_history": [],
        "expected": {
            "should_include": ["电话模拟", "脱敏训练"],
            "focus": "电话场景专项"
        }
    },
    {
        "id": "PLAN_010",
        "name": "综合提升计划",
        "user_profile": {
            "stuttering_type": "混合",
            "severity_level": 5,
            "trigger_factors": ["紧张", "快语速", "特定音"],
            "recommended_techniques": ["多技巧组合"],
            "available_time": "1小时/天",
            "level": "中级"
        },
        "training_history": [
            {"date": "2024-01", "type": "基础训练", "duration": "1个月"}
        ],
        "expected": {
            "should_include": ["技巧组合", "综合应用", "进阶挑战"],
            "duration_days": (14, 21)
        }
    }
]

# 汇总所有测试用例
TEST_CASES = {
    "expert_team": EXPERT_TEAM_TEST_CASES,
    "training_plan": TRAINING_PLAN_TEST_CASES,
}

# ========================================
# 评分标准说明
# ========================================
EVALUATION_CRITERIA = {
    "expert_team": {
        "路由准确性": """
评分标准:
- 10分: 完美识别问题类型，调用最合适的专家组合
- 7-9分: 正确识别主要类型，专家选择合理
- 4-6分: 类型识别基本正确，但专家选择有偏差
- 1-3分: 类型识别错误，专家完全不匹配
- 0分: 无法理解问题
""",
        "专家协作度": """
评分标准:
- 10分: 多专家意见整合完美，输出一致且全面
- 7-9分: 专家协作良好，输出协调
- 4-6分: 有协作但存在重复或遗漏
- 1-3分: 专家各自为战，输出混乱
- 0分: 完全无协作
""",
        "回答完整性": """
评分标准:
- 10分: 完全覆盖问题的所有层面
- 7-9分: 覆盖主要层面，略有遗漏
- 4-6分: 覆盖部分层面
- 1-3分: 仅覆盖一小部分
- 0分: 答非所问
""",
        "专业度": """
评分标准:
- 10分: 使用专业术语准确，建议科学合理
- 7-9分: 整体专业，小瑕疵可忽略
- 4-6分: 基本专业，有少量不严谨之处
- 1-3分: 专业性不足，建议可能有害
- 0分: 完全不专业
"""
    },
    "training_plan": {
        "计划可行性": """
评分标准:
- 10分: 计划完全符合用户时间/能力，可100%执行
- 7-9分: 计划可行，小调整即可
- 4-6分: 基本可行，需要较大调整
- 1-3分: 难以执行，不符合用户实际
- 0分: 完全不可行
""",
        "个性化程度": """
评分标准:
- 10分: 完全基于用户画像定制，无通用模板痕迹
- 7-9分: 高度个性化，少量通用内容
- 4-6分: 部分个性化，有模板痕迹
- 1-3分: 通用计划，个性化不足
- 0分: 完全通用
""",
        "目标明确性": """
评分标准:
- 10分: 每项目标都清晰、可量化、可达成
- 7-9分: 目标较明确，个别模糊
- 4-6分: 目标基本明确，部分不可量化
- 1-3分: 目标模糊，难以评估
- 0分: 无明确目标
""",
        "进阶合理性": """
评分标准:
- 10分: 难度递进完美，符合学习曲线
- 7-9分: 递进合理，偶有跳跃
- 4-6分: 递进基本合理，部分难度不当
- 1-3分: 难度混乱，跳跃过大或过小
- 0分: 无递进逻辑
"""
    }
}
