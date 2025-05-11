#!/usr/bin/env python3
"""英文字母和數字識別的 CNN 模型訓練腳本。"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    cast,
    runtime_checkable,
)

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# 型別別名定義
PathLike = Union[str, Path]
Tensor = torch.Tensor


@runtime_checkable
class SizedDataset(Protocol):
    """定義具有 __len__ 方法的資料集協定。"""

    def __len__(self) -> int: ...


# 標籤對照表
LABEL_MAP: Dict[str, int] = {
    # 數字 0-9
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    # 大寫字母 A-Z
    "A_caps": 10,
    "B_caps": 11,
    "C_caps": 12,
    "D_caps": 13,
    "E_caps": 14,
    "F_caps": 15,
    "G_caps": 16,
    "H_caps": 17,
    "I_caps": 18,
    "J_caps": 19,
    "K_caps": 20,
    "L_caps": 21,
    "M_caps": 22,
    "N_caps": 23,
    "O_caps": 24,
    "P_caps": 25,
    "Q_caps": 26,
    "R_caps": 27,
    "S_caps": 28,
    "T_caps": 29,
    "U_caps": 30,
    "V_caps": 31,
    "W_caps": 32,
    "X_caps": 33,
    "Y_caps": 34,
    "Z_caps": 35,
    # 小寫字母 a-z
    "a": 36,
    "b": 37,
    "c": 38,
    "d": 39,
    "e": 40,
    "f": 41,
    "g": 42,
    "h": 43,
    "i": 44,
    "j": 45,
    "k": 46,
    "l": 47,
    "m": 48,
    "n": 49,
    "o": 50,
    "p": 51,
    "q": 52,
    "r": 53,
    "s": 54,
    "t": 55,
    "u": 56,
    "v": 57,
    "w": 58,
    "x": 59,
    "y": 60,
    "z": 61,
}


# 常數定義
BASE_DIR = Path(__file__).parent / "data" / "handwriting"
TRAIN_CSV = BASE_DIR / "image_labels.csv"
TRAIN_DIR = BASE_DIR / "augmented_images" / "augmented_images1"
TEST_DIR = (
    BASE_DIR / "handwritten-english-characters-and-digits" / "combined_folder" / "test"
)
MODEL_PATH = BASE_DIR / "best_cnn.pth"

# 訓練參數
IMG_SIZE = 32
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_CLASSES = 62


class LettersDigitsDataset(Dataset[Tuple[Tensor, int]]):
    """英文字母和數字資料集類別。"""

    def __init__(
        self,
        csv_path: Optional[PathLike],
        img_dir: PathLike,
        transform: Optional[transforms.Compose] = None,
        is_test: bool = False,
    ) -> None:
        """初始化資料集。

        Args:
            csv_path: 標籤 CSV 檔案路徑（訓練時需要，測試時可為 None）
            img_dir: 圖片目錄路徑
            transform: 圖片轉換操作
            is_test: 是否為測試資料集
        """
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_test = is_test

        if is_test:
            # 測試時，直接讀取目錄中的所有圖片
            self.image_files = sorted([f for f in self.img_dir.glob("**/*.png")])
        else:
            # 訓練時，從 CSV 讀取標籤
            if csv_path is None:
                raise ValueError("訓練資料集需要提供標籤檔案")
            self.df = pd.read_csv(str(csv_path))

    def __len__(self) -> int:
        if self.is_test:
            return len(self.image_files)
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        if self.is_test:
            # 測試時，只返回圖片，標籤設為 -1
            img_path = self.image_files[idx]
            image = Image.open(img_path).convert("L")
        else:
            # 訓練時，從 CSV 讀取圖片路徑和標籤
            img_name = str(self.df.iloc[idx, 0])
            label_str = str(self.df.iloc[idx, 1])

            if label_str not in LABEL_MAP:
                raise ValueError(f"未知的標籤: {label_str}")
            label = LABEL_MAP[label_str]

            img_path = self.img_dir / img_name
            image = Image.open(img_path).convert("L")

        if self.transform:
            image_tensor = self.transform(image)
            if not isinstance(image_tensor, Tensor):
                raise TypeError("轉換後的圖片必須是 Tensor 類型")
            return image_tensor, (label if not self.is_test else -1)

        # 如果沒有轉換，手動轉換為 Tensor
        image_tensor = transforms.ToTensor()(image)
        return image_tensor, (label if not self.is_test else -1)


class SimpleCNN(nn.Module):
    """簡單的 CNN 模型用於字母和數字識別。"""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        """初始化 CNN 模型。

        Args:
            num_classes: 分類數量（預設為 62，包含 26 個字母和 10 個數字）
        """
        super().__init__()

        # 卷積層
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # 全連接層
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """前向傳播。

        Args:
            x: 輸入張量，形狀為 (batch_size, 1, height, width)

        Returns:
            輸出張量，形狀為 (batch_size, num_classes)
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


def get_transforms(img_size: int, is_train: bool = True) -> transforms.Compose:
    """獲取資料轉換操作。

    Args:
        img_size: 圖片大小
        is_train: 是否為訓練集轉換

    Returns:
        轉換操作組合
    """
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(10),
                transforms.RandomCrop(img_size, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """訓練一個 epoch。

    Args:
        model: CNN 模型
        loader: 訓練資料載入器
        criterion: 損失函數
        optimizer: 優化器
        device: 運算裝置

    Returns:
        平均訓練損失
    """
    model.train()
    total_loss = 0.0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(cast(SizedDataset, loader.dataset))


def eval_model(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> float:
    """評估模型。

    Args:
        model: CNN 模型
        loader: 評估資料載入器
        device: 運算裝置

    Returns:
        準確率
    """
    model.eval()
    correct = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

    return correct / len(cast(SizedDataset, loader.dataset))


def predict_model(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> List[int]:
    """使用模型進行預測。

    Args:
        model: CNN 模型
        loader: 測試資料載入器
        device: 運算裝置

    Returns:
        預測結果列表
    """
    model.eval()
    predictions: List[int] = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy().tolist())

    return predictions


def calculate_accuracy(predictions: List[int], true_labels: List[int]) -> float:
    """計算預測準確率。

    Args:
        predictions: 預測結果列表
        true_labels: 真實標籤列表

    Returns:
        準確率
    """
    correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    return correct / len(true_labels) if true_labels else 0.0


def main() -> None:
    """主函數。"""
    # 確保目錄存在
    TRAIN_DIR.parent.mkdir(parents=True, exist_ok=True)
    TEST_DIR.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 準備資料集和載入器
    train_dataset = LettersDigitsDataset(
        TRAIN_CSV,
        TRAIN_DIR,
        transform=get_transforms(IMG_SIZE, is_train=True),
        is_test=False,
    )
    test_dataset = LettersDigitsDataset(
        None,  # 測試時不需要標籤檔案
        TEST_DIR,
        transform=get_transforms(IMG_SIZE, is_train=False),
        is_test=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 初始化模型、損失函數和優化器
    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 訓練迴圈
    best_acc = 0.0
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {train_loss:.4f}")

    print("\n訓練完成！")

    # 儲存模型
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已儲存至: {MODEL_PATH}")

    # 對測試資料進行預測
    predictions = predict_model(model, test_loader, device)
    print(f"\n測試資料預測完成，共 {len(predictions)} 張圖片")

    # 從檔案名稱中提取真實標籤
    true_labels = []
    for img_path in test_dataset.image_files:
        # 假設檔案名稱格式為 "label_*.png"
        label_str = img_path.stem.split("_")[0]
        if label_str in LABEL_MAP:
            true_labels.append(LABEL_MAP[label_str])
        else:
            print(f"警告：無法從檔案名稱 {img_path.name} 提取標籤")
            true_labels.append(-1)  # 使用 -1 表示未知標籤

    # 計算準確率
    accuracy = calculate_accuracy(predictions, true_labels)
    print(f"\n測試準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    main()
