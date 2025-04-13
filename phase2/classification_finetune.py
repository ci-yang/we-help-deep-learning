from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models.doc2vec import Doc2Vec
from torch.utils.data import DataLoader, Dataset


@dataclass
class TextDataset(Dataset):
    embeddings: List[List[float]]
    labels: List[int]

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.embeddings[idx], dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class Classifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


@dataclass
class DataProcessor:
    csv_file: str
    doc2vec_model_path: str

    def load_csv_data(self) -> Tuple[List[List[str]], List[str]]:
        all_tokens = []
        all_labels = []
        print("📥 載入 CSV 資料...")

        with open(self.csv_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 讀取並分割 tokens 欄位
                tokens = row["tokens"].split()
                if tokens:  # 確保有 tokens
                    all_tokens.append(tokens)
                    all_labels.append(row["board"])

        if not all_tokens:
            raise ValueError("沒有找到有效的數據！請檢查 CSV 檔案格式。")

        return all_tokens, all_labels

    def load_doc2vec_model(self) -> Tuple[Any, int]:
        print("📥 載入預訓練的 Doc2Vec 模型...")
        d2v_model = Doc2Vec.load(self.doc2vec_model_path)
        return d2v_model, d2v_model.vector_size

    def get_embeddings(
        self, all_tokens: List[List[str]], d2v_model: Doc2Vec, epochs: int = 50
    ) -> List[List[float]]:
        print("🔄 產生每個文件的 embeddings...")
        embeddings = []
        vector_size = d2v_model.vector_size
        zero_vector = np.zeros(vector_size)  # 預設的零向量

        for tokens in all_tokens:
            # 只保留在詞彙表中的詞
            valid_words = [word for word in tokens if word in d2v_model.wv]

            if valid_words:
                # 如果有有效的詞，計算平均值
                embedding = np.mean(
                    [d2v_model.wv[word] for word in valid_words], axis=0
                )
            else:
                # 如果沒有有效的詞，使用零向量
                embedding = zero_vector

            embeddings.append(embedding.tolist())

        return embeddings


@dataclass
class ModelTrainer:
    hidden_dim: int = 64
    num_epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 0.01
    train_ratio: float = 0.8
    patience: int = 10  # 停止訓練的耐心值，如果連續 patience 個 epoch 沒有改善就停止

    def encode_labels(self, label_list: List[str]) -> Tuple[List[int], Dict[str, int]]:
        label_set = sorted(set(label_list))
        label2idx = {label: idx for idx, label in enumerate(label_set)}
        encoded = [label2idx[label] for label in label_list]
        return encoded, label2idx

    def split_data(
        self, embeddings: List[List[float]], labels: List[int]
    ) -> Tuple[List, List]:
        data = list(zip(embeddings, labels))
        random.shuffle(data)
        split_idx = int(self.train_ratio * len(data))
        train_data = data[:split_idx]
        test_data = data[split_idx:]

        print(f"\n📊 資料分割:")
        print(f"訓練資料: {len(train_data)} 筆 ({self.train_ratio*100:.0f}%)")
        print(f"測試資料: {len(test_data)} 筆 ({(1-self.train_ratio)*100:.0f}%)")

        return train_data, test_data

    def create_dataloaders(
        self, train_data: List, test_data: List
    ) -> Tuple[DataLoader, DataLoader]:
        train_embeddings, train_labels = zip(*train_data)
        test_embeddings, test_labels = zip(*test_data)

        train_dataset = TextDataset(train_embeddings, train_labels)
        test_dataset = TextDataset(test_embeddings, test_labels)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        return train_loader, test_loader

    def evaluate_model(
        self, model: Classifier, data_loader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, float, float]:
        model.eval()
        total_loss = 0.0
        total_samples = 0
        correct_top1 = 0
        correct_top2 = 0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                batch_size = batch_x.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                _, top1 = torch.max(outputs, 1)
                correct_top1 += (top1 == batch_y).sum().item()

                _, top2 = torch.topk(outputs, k=2, dim=1)
                correct_top2 += sum(batch_y[i] in top2[i] for i in range(batch_size))

        return (
            total_loss / total_samples,
            correct_top1 / total_samples,
            correct_top2 / total_samples,
        )

    def train_and_evaluate(
        self,
        model: Classifier,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> None:
        best_test_acc = 0.0
        no_improve_epochs = 0  # 記錄沒有改善的 epoch 數

        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0.0
            total_samples = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)

            train_loss = running_loss / total_samples
            test_loss, top1_acc, top2_acc = self.evaluate_model(
                model, test_loader, criterion
            )

            print(f"\n===== Epoch {epoch+1}/{self.num_epochs} =====")
            print(f"📊 訓練資料平均損失: {train_loss:.6f}")
            print(f"📊 測試資料平均損失: {test_loss:.6f}")
            print(f"🎯 Top-1 準確率: {top1_acc:.6f}")
            print(f"🎯 Top-2 準確率: {top2_acc:.6f}")

            # 檢查是否有改善
            if top1_acc > best_test_acc:
                best_test_acc = top1_acc
                no_improve_epochs = 0  # 重置計數器
            else:
                no_improve_epochs += 1
                print(f"⚠️ 連續 {no_improve_epochs} 個 epoch 沒有改善")

            # 檢查是否需要停止訓練
            if no_improve_epochs >= self.patience:
                print(f"\n🛑 連續 {self.patience} 個 epoch 沒有改善，停止訓練")
                print(f"🎯 最佳 Top-1 準確率: {best_test_acc:.6f}")
                break

        if no_improve_epochs < self.patience:
            print(f"\n✅ 訓練完成！")
            print(f"🎯 最佳 Top-1 準確率: {best_test_acc:.6f}")


def execute_training():
    random.seed(42)
    torch.manual_seed(42)

    # 設定參數
    CSV_FILE = "data/tokenized_data.csv"
    DOC2VEC_MODEL_PATH = "doc2vec_model.model"

    # 初始化處理器和訓練器
    processor = DataProcessor(CSV_FILE, DOC2VEC_MODEL_PATH)
    trainer = ModelTrainer()

    # 載入和處理資料
    all_tokens, all_labels = processor.load_csv_data()
    print(f"📝 總文件數: {len(all_tokens)}")

    encoded_labels, label2idx = trainer.encode_labels(all_labels)
    num_classes = len(label2idx)
    print(f"📊 分類類別數: {num_classes}")

    # 載入模型和產生 embeddings
    d2v_model, vector_size = processor.load_doc2vec_model()
    embeddings = processor.get_embeddings(all_tokens, d2v_model, epochs=50)

    # 準備訓練資料
    train_data, test_data = trainer.split_data(embeddings, encoded_labels)
    train_loader, test_loader = trainer.create_dataloaders(train_data, test_data)

    # 建立和訓練模型
    print("🚀 開始訓練分類器...")
    classifier = Classifier(vector_size, trainer.hidden_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=trainer.learning_rate)

    trainer.train_and_evaluate(
        classifier, train_loader, test_loader, criterion, optimizer
    )


if __name__ == "__main__":
    execute_training()
