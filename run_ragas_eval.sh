#!/bin/bash
# RAGAS 评估一键运行脚本

echo "🚀 启动 RAGAS 评估..."

# 1. 检查并启动 Python 向量库服务
echo "📦 检查向量库服务..."
if ! curl -s http://localhost:8002/health > /dev/null; then
    echo "  启动向量库服务..."
    nohup python3 memory_service_only.py > /tmp/memory_service.log 2>&1 &
    sleep 3
fi
echo "  ✅ 向量库服务就绪"

# 2. 检查并启动 Go 后端
echo "🔧 检查 Go 后端..."
if ! curl -s http://localhost:8001/api/v1/ai/available-roles > /dev/null; then
    echo "  启动 Go 后端..."
    cd /Users/zhangxiaobin/self-project/fluent-life-api
    export PYTHON_AI_SERVICE_URL=http://localhost:8002
    nohup go run ./cmd/server/main.go > /tmp/go_service.log 2>&1 &
    sleep 5
fi
echo "  ✅ Go 后端就绪"

# 3. 运行 RAGAS 评估
echo ""
echo "📊 开始 RAGAS 评估..."
cd /Users/zhangxiaobin/self-project/fluent-life-ai-service
export GO_API_URL=http://localhost:8001
export USE_GO_API=true
python3 -m evaluation.ragas_memory_evaluator

echo ""
echo "📄 查看报告: cat evaluation/reports/ragas_memory_report.json"
