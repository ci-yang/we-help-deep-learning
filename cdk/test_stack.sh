#!/bin/bash

# 設置環境變數（如果沒有設置的話）
if [ -z "$CDK_DEFAULT_ACCOUNT" ]; then
    echo "請設置 AWS 帳號 ID:"
    read AWS_ACCOUNT_ID
    export CDK_DEFAULT_ACCOUNT=$AWS_ACCOUNT_ID
fi

if [ -z "$CDK_DEFAULT_REGION" ]; then
    export CDK_DEFAULT_REGION="ap-southeast-1"
fi

# 顯示當前配置
echo "使用以下配置:"
echo "AWS Account: $CDK_DEFAULT_ACCOUNT"
echo "AWS Region: $CDK_DEFAULT_REGION"

# 安裝依賴
pip install -r requirements.txt

# CDK 合成測試
echo "執行 CDK 合成測試..."
cdk synth

# 如果合成成功，執行 diff
if [ $? -eq 0 ]; then
    echo "合成成功，執行 diff..."
    cdk diff
else
    echo "合成失敗，請檢查錯誤"
    exit 1
fi

# 詢問是否要進行部署
read -p "是否要進行部署測試？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "執行部署..."
    cdk deploy --hotswap
fi