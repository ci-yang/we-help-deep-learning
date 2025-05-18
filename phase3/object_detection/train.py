import os

import torch
import torchvision.transforms as T
from dataset import VehicleDataset
from model import get_model, train_one_epoch
from torch.optim import SGD
from torch.utils.data import DataLoader


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    # 設定參數
    num_epochs = 10
    batch_size = 4
    learning_rate = 0.005
    momentum = 0.9
    weight_decay = 0.0005

    # 設定裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 資料轉換
    transform = T.Compose(
        [
            T.ToTensor(),
        ]
    )

    # 載入資料集
    train_dataset = VehicleDataset(
        image_dir="data/vehicles_images/train",
        csv_path="data/vehicles_images/train_labels.csv",
        transform=transform,
    )

    # 建立資料載入器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # 初始化模型
    num_classes = len(train_dataset.categories) + 1  # +1 為背景類別
    model = get_model(num_classes)
    model.to(device)

    # 設定優化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(
        params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )

    # 訓練模型
    print("開始訓練...")
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # 儲存模型
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/faster_rcnn_model.pth")
    print("模型已儲存至 checkpoints/faster_rcnn_model.pth")


if __name__ == "__main__":
    main()
