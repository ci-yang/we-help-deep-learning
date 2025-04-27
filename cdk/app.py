#!/usr/bin/env python3
import os

from aws_cdk import App

from cdk.cdk_stack import CdkStack

app = App()
CdkStack(
    app,
    "FastAPIDeploymentStack",
    env={
        "account": os.getenv("CDK_DEFAULT_ACCOUNT"),
        "region": os.getenv("CDK_DEFAULT_REGION"),
    },
)
app.synth()
