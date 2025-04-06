from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models.doc2vec import Doc2Vec
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class TextDataset(Dataset):
    embeddings: torch.Tensor
    labels: torch.Tensor

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


class ClassificationModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        return self.layer2(x)


@dataclass
class ModelTrainer:
    model: ClassificationModel
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 300
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    criterion: nn.CrossEntropyLoss = field(init=False)
    optimizer: optim.AdamW = field(init=False)
    scheduler: optim.lr_scheduler.OneCycleLR = field(init=False)
    class_weights: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.model = self.model.to(self.device)
        self.class_weights = processor.class_weights.to(self.device)  # type: ignore
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights, label_smoothing=0.1
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )

    def _init_scheduler(self, num_training_steps: int):
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=num_training_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(embeddings)
            loss = self.criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        num_classes = self.model.layer2[-1].out_features  # 直接訪問 out_features
        class_correct = [0] * num_classes  # type: ignore
        class_total = [0] * num_classes  # type: ignore

        with torch.no_grad():
            for embeddings, labels in test_loader:
                embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

                # 計算每個類別的準確率
                for label, pred in zip(labels.cpu(), predicted.cpu()):
                    label_idx = int(label.item())  # type: ignore
                    if label == pred:
                        class_correct[label_idx] += 1
                    class_total[label_idx] += 1

        # 輸出每個類別的準確率
        print("\n📊 各類別準確率:")
        for i in range(num_classes):  # type: ignore
            accuracy = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            print(f"類別 {i}: {accuracy:.4f}")

        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(test_loader)
        return accuracy, avg_loss

    def train(
        self, train_loader: DataLoader, test_loader: DataLoader
    ) -> Dict[str, List[float]]:
        # 初始化學習率調度器
        total_steps = len(train_loader) * self.num_epochs
        self._init_scheduler(total_steps)

        history = {"train_loss": [], "test_loss": [], "test_accuracy": []}

        best_accuracy = 0.0
        patience = 30  # 較短的耐心值
        no_improve = 0

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(train_loader)
            test_accuracy, test_loss = self.evaluate(test_loader)

            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["test_accuracy"].append(test_accuracy)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                no_improve = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                no_improve += 1

            print(f"Epoch [{epoch+1}/{self.num_epochs}]")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Best Accuracy: {best_accuracy:.4f}")
            print("------------------------")

            if no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # 載入最佳模型
        self.model.load_state_dict(torch.load("best_model.pth"))
        return history


class DataProcessor:
    def __init__(self, doc2vec_model: Doc2Vec, data_path: str):
        self.doc2vec_model = doc2vec_model
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        self.label_encoder = LabelEncoder()

        # 分析類別分布
        self.analyze_class_distribution()

        # 將 Board 欄位轉換為數值標籤
        self.data["label"] = self.label_encoder.fit_transform(self.data["Board"])

        # 設置類別數量
        self.num_classes = len(self.label_encoder.classes_)

        # 計算類別權重
        self.calculate_class_weights()

    def calculate_class_weights(self):
        """計算類別權重以處理不平衡數據"""
        label_counts = self.data["Board"].value_counts()
        total_samples = len(self.data)
        self.class_weights = torch.FloatTensor(
            [
                total_samples / (len(label_counts) * count)
                for count in label_counts.values
            ]
        )

    def analyze_class_distribution(self):
        """分析並顯示類別分布"""
        class_counts = self.data["Board"].value_counts()
        total = len(self.data)
        print("\n📊 類別分布:")
        for board, count in class_counts.items():
            percentage = (count / total) * 100
            print(f"{board}: {count} ({percentage:.2f}%)")

    def create_data_loaders(
        self, batch_size: int = 64
    ) -> Tuple[DataLoader, DataLoader]:
        """創建訓練和測試數據加載器"""
        # 將標題轉換為向量
        embeddings = []
        for title in self.data["Title"]:
            vector = self.doc2vec_model.infer_vector(title.split())
            embeddings.append(vector)

        # 使用 numpy 數組優化張量創建
        embeddings_array = np.array(embeddings)

        # 創建數據集
        dataset = TextDataset(
            embeddings=torch.tensor(embeddings_array, dtype=torch.float32),
            labels=torch.tensor(self.data["label"].values, dtype=torch.long),
        )

        # 分割訓練和測試集
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # 創建數據加載器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader


def execute_training() -> None:
    print("📥 載入 Doc2Vec 模型...")
    doc2vec_model: Doc2Vec = Doc2Vec.load("doc2vec_original_optimized.model")  # type: ignore

    print("🔄 準備數據...")
    global processor
    processor = DataProcessor(
        doc2vec_model=doc2vec_model, data_path="data/cleaned_data.csv"
    )
    train_loader, test_loader = processor.create_data_loaders(batch_size=64)

    print("🏗️ 建立分類模型...")
    model = ClassificationModel(
        input_size=doc2vec_model.vector_size,
        hidden_size=256,  # 減小隱藏層
        num_classes=processor.num_classes,
    )

    print(f"類別數量: {processor.num_classes}")
    print("🚀 開始訓練模型...")
    trainer = ModelTrainer(model=model)
    history = trainer.train(train_loader, test_loader)

    final_accuracy = history["test_accuracy"][-1]
    print(f"最終測試準確率: {final_accuracy:.4f}")


if __name__ == "__main__":
    execute_training()
