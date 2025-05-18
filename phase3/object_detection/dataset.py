import os

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class VehicleDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        """
        初始化資料集
        Args:
            image_dir (str): 圖片目錄路徑
            csv_path (str): 標註檔案路徑
            transform (callable, optional): 圖片轉換
        """
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        self.transform = transform or T.Compose(
            [
                T.ToTensor(),
            ]
        )

        # 讀取類別對應
        with open(os.path.join(os.path.dirname(csv_path), "category.txt"), "r") as f:
            content = f.read().strip()
            # 移除 'category: ' 前綴並解析字典
            dict_str = content.replace("category: ", "")
            # 使用 eval 解析字典，但確保順序正確
            raw_mapping = eval(dict_str)
            # 按照原始字典的順序建立類別列表
            self.categories = ["Bus", "Car", "Motorcycle", "Pickup", "Truck"]
            # 使用原始映射值
            self.category_to_idx = raw_mapping

    def __len__(self):
        return len(self.df["filename"].unique())

    def __getitem__(self, idx):
        # 獲取圖片 ID
        image_id = str(self.df["filename"].unique()[idx])

        # 載入圖片
        img_path = os.path.join(self.image_dir, image_id)
        image = Image.open(img_path).convert("RGB")

        # 獲取該圖片的所有標註
        img_annotations = self.df[self.df["filename"] == image_id]

        # 準備 bounding boxes 和 labels
        boxes = []
        labels = []

        for _, row in img_annotations.iterrows():
            # 獲取 bounding box 座標
            xmin = row["xmin"]
            ymin = row["ymin"]
            xmax = row["xmax"]
            ymax = row["ymax"]

            # 獲取類別標籤
            category = row["class"]
            label = self.category_to_idx[category]

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        # 轉換為 tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # 應用轉換
        if self.transform:
            image = self.transform(image)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        return image, target
