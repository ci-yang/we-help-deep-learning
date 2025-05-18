#!/usr/bin/env python3
"""載入已訓練的 CNN 模型並評估測試資料準確率。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# 型別別名定義
PathLike = Union[str, Path]
Tensor = torch.Tensor

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
TEST_DIR = (
    BASE_DIR / "handwritten-english-characters-and-digits" / "combined_folder" / "test"
)
MODEL_PATH = BASE_DIR / "best_cnn.pth"
IMG_SIZE = 32
BATCH_SIZE = 128
NUM_CLASSES = 62


class SimpleCNN(nn.Module):
    """簡單的 CNN 模型用於字母和數字識別。"""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


class TestDataset(Dataset[Tuple[Tensor, int]]):
    """測試資料集類別。"""

    def __init__(self, img_dir: PathLike) -> None:
        self.img_dir = Path(img_dir)
        self.image_files = []
        self.labels = []

        # 遍歷每個類別目錄
        for class_dir in sorted(self.img_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name.startswith("."):
                continue

            # 從目錄名稱獲取標籤
            label_str = class_dir.name
            if label_str not in LABEL_MAP:
                print(f"警告：未知的類別目錄 {label_str}")
                continue

            # 收集該類別的所有圖片
            for img_path in sorted(class_dir.glob("*.png")):
                self.image_files.append(img_path)
                self.labels.append(LABEL_MAP[label_str])

        self.transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("L")
        image_tensor = self.transform(image)
        if not isinstance(image_tensor, Tensor):
            raise TypeError("轉換後的圖片必須是 Tensor 類型")

        return image_tensor, self.labels[idx]


def predict_model(
    model: nn.Module,
    loader: DataLoader[Tuple[Tensor, int]],
    device: torch.device,
) -> List[int]:
    """使用模型進行預測。"""
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
    """計算預測準確率。"""
    correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    return correct / len(true_labels) if true_labels else 0.0


def main() -> None:
    """主函數。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 準備測試資料集
    test_dataset = TestDataset(TEST_DIR)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    # 載入模型
    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print(f"已載入模型: {MODEL_PATH}")

    # 進行預測
    predictions = predict_model(model, test_loader, device)
    true_labels = [label for _, label in test_dataset]

    # 計算整體準確率
    accuracy = calculate_accuracy(predictions, true_labels)
    print(f"\n測試準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    main()
