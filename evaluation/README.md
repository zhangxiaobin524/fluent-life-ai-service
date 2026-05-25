# LLM-as-a-Judge 评测系统

使用 DeepSeek API 作为裁判，自动评估 AI 模块输出质量。

## 快速开始

### 1. 安装依赖

```bash
# 确保已安装依赖（在 fluent-life-ai-service 目录）
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
export DEEPSEEK_API_KEY=sk-your-api-key
```

### 3. 快速测试（Mock 模式）

```bash
python -m evaluation.quick_test
```

这会使用模拟的 AI 输出测试评测系统，验证评测逻辑是否正确。

### 4. 真实评测

```bash
python -m evaluation.run_evaluation
```

这会调用真实的 ExpertTeam 和 TrainingPlan 工作流进行评测。

## 测试用例结构

### ExpertTeam (10个测试用例)

| ID | 名称 | 类型 | 预期专家 |
|----|------|------|---------|
| EXP_001 | 心理障碍咨询 | psychological | 心理支持专家 |
| EXP_002 | 训练数据分析 | data_analysis | 数据分析师 |
| EXP_003 | 训练计划制定 | training_plan | 方案专家 |
| EXP_004 | 症状诊断 | diagnosis | 诊断专家 |
| EXP_005 | 复杂问题（多维度） | complex | 多个专家 |
| EXP_006 | 简单问候 | general | 直接回答 |
| EXP_007 | 技巧询问 | training_plan | 方案专家 |
| EXP_008 | 情绪支持 | psychological | 心理支持专家 |
| EXP_009 | 效果对比 | data_analysis | 数据分析师 |
| EXP_010 | 触发场景咨询 | diagnosis | 诊断专家 |

### TrainingPlan (10个测试用例)

| ID | 名称 | 用户特征 |
|----|------|---------|
| PLAN_001 | 初级用户基础计划 | 初级，首字难发 |
| PLAN_002 | 中级用户进阶计划 | 中级，重复发音 |
| PLAN_003 | 高级用户维持计划 | 高级，间歇性 |
| PLAN_004 | 时间紧张用户计划 | 每天15分钟 |
| PLAN_005 | 严重口吃康复计划 | 重度，多重症状 |
| PLAN_006 | 儿童用户专项计划 | 8岁，发育性 |
| PLAN_007 | 职场人士实战计划 | 销售，情境性 |
| PLAN_008 | 反馈调整测试 | 难度反馈 |
| PLAN_009 | 特定场景突破计划 | 电话恐惧 |
| PLAN_010 | 综合提升计划 | 混合类型 |

## 评测维度

### ExpertTeam 评测维度

| 维度 | 权重 | 说明 |
|------|------|------|
| 路由准确性 | 30% | 正确识别问题类型，调用合适的专家 |
| 专家协作度 | 30% | 多专家意见整合协调 |
| 回答完整性 | 20% | 覆盖问题的各个层面 |
| 专业度 | 20% | 口吃矫正领域的专业程度 |

### TrainingPlan 评测维度

| 维度 | 权重 | 说明 |
|------|------|------|
| 计划可行性 | 30% | 用户能够实际执行 |
| 个性化程度 | 30% | 符合用户个人情况 |
| 目标明确性 | 20% | 目标清晰可量化 |
| 进阶合理性 | 20% | 难度递进合理 |

## 评分标准

- **10分**: 完美，无可挑剔
- **7-9分**: 良好，有小瑕疵
- **4-6分**: 一般，有明显问题
- **1-3分**: 较差，需要大幅改进
- **0分**: 完全不合格

## 报告输出

评测完成后会生成 Markdown 报告，包含：

- 各模块综合得分
- 各维度详细评分
- 低分项目列表及改进建议
- 原始评测数据

## 代码结构

```
evaluation/
├── __init__.py          # 模块导出
├── test_cases.py        # 测试用例定义
├── evaluator.py         # 评测核心逻辑
├── run_evaluation.py    # 完整评测脚本
├── quick_test.py        # 快速测试脚本
└── README.md            # 本文档
```

## 进阶使用

### 自定义测试用例

```python
from evaluation import TEST_CASES

# 添加新的测试用例
TEST_CASES["expert_team"].append({
    "id": "EXP_011",
    "name": "自定义测试",
    "user_message": "你的测试问题",
    "expected": {
        "question_type": "diagnosis",
        "should_involve": ["诊断专家"]
    }
})
```

### 自定义评测维度

```python
from evaluation import EVALUATION_DIMENSIONS

# 修改权重
EVALUATION_DIMENSIONS["expert_team"]["路由准确性"] = 40
```

### 使用自己的 AI 调用

```python
from evaluation import AIEvaluator, TEST_CASES

async def my_ai_call(test_case):
    # 调用你自己的 AI
    result = await my_model.predict(test_case)
    return result

evaluator = AIEvaluator()
result = evaluator.evaluate_module(
    "expert_team",
    TEST_CASES["expert_team"],
    my_ai_call
)
```

## 注意事项

1. **API 费用**: 每次评测需要调用 DeepSeek API，10个测试用例 × 4个维度 = 40次调用
2. **评分一致性**: 由于 LLM 有一定随机性，相同输入可能得到略微不同的分数
3. **温度设置**: 评测时使用 temperature=0.3 保证一定的稳定性
4. **Mock 模式**: `quick_test.py` 不调用真实工作流，仅用于测试评测系统本身

## 后续优化方向

1. 增加更多测试用例覆盖边界情况
2. 引入人工标注数据校准评分
3. 实现评测结果的趋势分析（历史对比）
4. 增加自动回归测试（CI/CD 集成）
