# type: ignore
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm


def get_model(num_classes):
    """
    初始化 Faster R-CNN 模型
    Args:
        num_classes (int): 類別數量（包含背景）
    Returns:
        model: Faster R-CNN 模型
    """
    # 載入預訓練模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # 修改分類器以適應我們的類別數量
    box_predictor = model.roi_heads.box_predictor
    in_features = box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_one_epoch(model, optimizer, data_loader, device):
    """
    訓練一個 epoch
    Args:
        model: Faster R-CNN 模型
        optimizer: 優化器
        data_loader: 資料載入器
        device: 訓練裝置
    Returns:
        float: 平均損失
    """
    model.train()
    total_loss = 0

    # 使用 tqdm 顯示進度條
    pbar = tqdm(data_loader, desc="Training")

    for images, targets in pbar:
        # 將資料移至指定裝置
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 計算損失
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # 反向傳播
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # 更新總損失
        total_loss += losses.item()

        # 更新進度條
        pbar.set_postfix({"loss": losses.item()})

    return total_loss / len(data_loader)


def evaluate(model, images, device):
    """評估模型"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for image in images:
            prediction = model([image])
            predictions.extend(prediction)

    return predictions
