import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from dataset import VehicleDataset
from model import evaluate, get_model
from torch.utils.data import DataLoader


def calculate_iou(box1, box2):
    """計算兩個邊界框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0


def calculate_metrics(predictions, targets, iou_threshold=0.5):
    """計算準確率指標"""
    total_correct = 0
    total_predictions = 0
    total_targets = 0

    for pred, target in zip(predictions, targets):
        pred_boxes = pred["boxes"].cpu().numpy()
        pred_labels = pred["labels"].cpu().numpy()
        pred_scores = pred["scores"].cpu().numpy()

        target_boxes = target["boxes"].cpu().numpy()
        target_labels = target["labels"].cpu().numpy()

        # 只考慮信心度大於 0.5 的預測
        mask = pred_scores > 0.5
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]

        # 計算每個預測框與目標框的 IoU
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_target_idx = -1

            for i, (target_box, target_label) in enumerate(
                zip(target_boxes, target_labels)
            ):
                iou = calculate_iou(pred_box, target_box)
                if iou > best_iou and pred_label == target_label:
                    best_iou = iou
                    best_target_idx = i

            if best_iou >= iou_threshold and best_target_idx != -1:
                total_correct += 1
                # 移除已匹配的目標框
                target_boxes = np.delete(target_boxes, best_target_idx, axis=0)
                target_labels = np.delete(target_labels, best_target_idx, axis=0)

        total_predictions += len(pred_boxes)
        total_targets += len(target["boxes"])

    precision = total_correct / total_predictions if total_predictions > 0 else 0
    recall = total_correct / total_targets if total_targets > 0 else 0
    accuracy = (precision + recall) / 2

    return {"precision": precision, "recall": recall, "accuracy": accuracy}


def visualize_predictions(image, predictions, save_path, categories):
    """視覺化預測結果"""
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    # 繪製預測框
    boxes = predictions["boxes"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()

    # 只顯示信心度大於 0.5 的預測
    mask = scores > 0.5
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]

    # 使用與訓練時相同的標籤映射
    label_mapping = {0: "Bus", 1: "Car", 2: "Motorcycle", 3: "Pickup", 4: "Truck"}

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        # 直接使用標籤映射
        category_name = label_mapping.get(label, f"Unknown-{label}")
        plt.text(
            x1,
            y1 - 5,
            f"{category_name} ({score:.2f})",
            color="white",
            bbox=dict(facecolor="red", alpha=0.5),
        )

    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    # 設定裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 載入測試資料集
    test_dataset = VehicleDataset(
        image_dir="data/vehicles_images/test",
        csv_path="data/vehicles_images/test_labels.csv",
        transform=T.Compose([T.ToTensor()]),
    )

    # 建立資料載入器
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    # 載入模型
    model = get_model(num_classes=len(test_dataset.categories) + 1)
    model.load_state_dict(torch.load("checkpoints/faster_rcnn_model.pth"))
    model.to(device)
    model.eval()

    # 評估模型
    print("開始評估...")
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            predictions = evaluate(model, images, device)
            all_predictions.extend(predictions)
            all_targets.extend(targets)

    # 計算準確率
    metrics = calculate_metrics(all_predictions, all_targets)
    print(f"\n評估結果:")
    print(f'精確率 (Precision): {metrics["precision"]:.2%}')
    print(f'召回率 (Recall): {metrics["recall"]:.2%}')
    print(f'準確率 (Accuracy): {metrics["accuracy"]:.2%}')

    if metrics["accuracy"] < 0.6:
        print("\n警告：準確率未達到 60% 的要求！")
    else:
        print("\n恭喜！準確率已達到要求。")

    # 視覺化預測結果
    os.makedirs("predictions", exist_ok=True)
    for i, (images, _) in enumerate(test_loader):
        img = images[0].permute(1, 2, 0).cpu().numpy()
        save_path = f"predictions/pred_{i}.jpg"
        visualize_predictions(
            img, all_predictions[i], save_path, test_dataset.categories
        )

    print(f"\n預測結果已儲存至 predictions 目錄")


if __name__ == "__main__":
    main()
