from aws_cdk import Stack
from aws_cdk import aws_ec2 as ec2  # Duration,; aws_sqs as sqs,
from aws_cdk import aws_iam as iam
from constructs import Construct


class CdkStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # The code that defines your stack goes here

        # example resource
        # queue = sqs.Queue(
        #     self, "CdkQueue",
        #     visibility_timeout=Duration.seconds(300),
        # )

        # 建立 VPC
        vpc = ec2.Vpc(self, "FastAPIVPC", max_azs=1, nat_gateways=0)

        # 建立安全群組
        security_group = ec2.SecurityGroup(
            self,
            "FastAPISecurityGroup",
            vpc=vpc,
            description="Security group for FastAPI application",
            allow_all_outbound=True,
        )

        # 允許 HTTP 流量
        security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(), ec2.Port.tcp(8000), "Allow HTTP traffic"
        )

        # 允許 SSH 流量
        security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(), ec2.Port.tcp(22), "Allow SSH traffic"
        )

        # 建立 IAM 角色
        role = iam.Role(
            self,
            "FastAPIRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com").grant_principal,
        )

        # 添加必要的權限
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonSSMManagedInstanceCore"
            )
        )

        # 使用者資料腳本
        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "#!/bin/bash",
            "exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1",
            "echo '開始安裝...'",
            "sudo yum update -y",
            "sudo yum install -y python3-pip git",
            "echo '安裝 poetry...'",
            "pip3 install poetry",
            "echo '克隆倉庫...'",
            "git clone https://github.com/ci-yang/we-help-deep-learning.git",
            "cd we-help-deep-learning/prediction_app",
            "echo '安裝依賴...'",
            "poetry install",
            "echo '啟動應用程式...'",
            "nohup poetry run python run.py > app.log 2>&1 &",
            "echo '應用程式已啟動'",
            "sleep 10",
            "echo '檢查應用程式狀態...'",
            "curl http://localhost:8000/docs || echo '應用程式未正常啟動'",
        )

        # 建立 EC2 實例
        instance = ec2.Instance(
            self,
            "FastAPIInstance2",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            instance_type=ec2.InstanceType("t2.micro"),
            machine_image=ec2.AmazonLinuxImage(
                generation=ec2.AmazonLinuxGeneration.AMAZON_LINUX_2
            ),
            security_group=security_group,
            role=iam.Role.from_role_arn(self, "ImportedRole", role.role_arn),
            user_data=user_data,
        )

        # 輸出實例的公共 IP
        self.instance_public_ip = instance.instance_public_ip
