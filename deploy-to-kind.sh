#!/bin/bash
# ============================================
# AI 服务部署到 Kind K8s 脚本（Docker Hub 版）
# ============================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  AI 服务部署到 Kind K8s${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查 kubectl
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}错误: kubectl 未安装${NC}"
    exit 1
fi

# ========================================
# Docker Hub 配置 - 修改为你的账号
# ========================================
DOCKERHUB_USERNAME="dockerzxb"  # 你的 Docker Hub 用户名
IMAGE_NAME="fluent-life-ai-service"
IMAGE_TAG="latest"

# 完整镜像地址
REMOTE_IMAGE="${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
LOCAL_IMAGE="fluent-life-ai-service:latest"

NAMESPACE="fluent-life"

cd /Users/zhangxiaobin/self-project/fluent-life-ai-service

# 步骤1: 构建镜像
echo -e "\n${YELLOW}[1/5] 构建 AI 服务镜像...${NC}"
docker build -t ${LOCAL_IMAGE} .
echo -e "${GREEN}✓ 镜像构建完成${NC}"

# 步骤2: 推送到 Docker Hub
echo -e "\n${YELLOW}[2/5] 推送到 Docker Hub...${NC}"

# 检查是否已登录 Docker Hub
if ! docker info 2>/dev/null | grep -q "Username"; then
    echo -e "${YELLOW}提示: 请先登录 Docker Hub${NC}"
    echo -e "执行: docker login"
    docker login
fi

# 打标签并推送
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE}
docker push ${REMOTE_IMAGE}
echo -e "${GREEN}✓ 镜像推送完成: ${REMOTE_IMAGE}${NC}"

# 步骤3: 更新 K8s 部署文件使用远程镜像
echo -e "\n${YELLOW}[3/5] 更新 K8s 部署配置...${NC}"

# 创建临时部署文件，使用远程镜像地址
sed "s|image: fluent-life-ai-service:latest|image: ${REMOTE_IMAGE}|g" k8s-deployment.yaml > k8s-deployment-remote.yaml

echo -e "${GREEN}✓ K8s 配置更新完成${NC}"

# 步骤4: 部署到 K8s
echo -e "\n${YELLOW}[4/5] 部署 AI 服务到 K8s...${NC}"
kubectl apply -f k8s-deployment-remote.yaml
echo -e "${GREEN}✓ 部署完成${NC}"

# 步骤5: 等待部署完成
echo -e "\n${YELLOW}[5/5] 等待 Pod 启动...${NC}"
kubectl wait --for=condition=ready pod -l app=ai-service -n ${NAMESPACE} --timeout=120s || true

# 显示状态
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}部署完成！${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\n${YELLOW}Pod 状态:${NC}"
kubectl get pods -n ${NAMESPACE} -l app=ai-service

echo -e "\n${YELLOW}Service 状态:${NC}"
kubectl get svc -n ${NAMESPACE} -l app=ai-service

echo -e "\n${YELLOW}镜像地址:${NC}"
echo -e "  ${REMOTE_IMAGE}"

echo -e "\n${YELLOW}访问方式:${NC}"
echo -e "  - K8s 内部访问: http://ai-service.${NAMESPACE}.svc.cluster.local:8002"
echo -e "  - 同命名空间访问: http://ai-service:8002"

echo -e "\n${YELLOW}查看日志:${NC}"
echo -e "  kubectl logs -n ${NAMESPACE} -l app=ai-service -f"

echo -e "\n${YELLOW}测试健康检查:${NC}"
echo -e "  kubectl exec -n ${NAMESPACE} deploy/ai-service -- wget -qO- http://localhost:8002/health"

# 清理临时文件
rm -f k8s-deployment-remote.yaml
