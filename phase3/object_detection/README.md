# Vehicle Detection with Faster R-CNN

## 資料下載

1. 前往以下連結下載資料集：
   [Google Drive 下載連結](https://drive.google.com/file/d/1xiIrAL9J5RFEHDRAZShTyzzTn_wa9ySk/view)
2. 解壓縮後，將資料夾放到 `phase3/object_detection/data` 目錄下。

## 環境安裝

建議使用 Python 3.8+。

```bash
pip install -r requirements.txt
```

## 訓練模型

```bash
python train.py
```
訓練完成後，模型權重會儲存在 `checkpoints/faster_rcnn_model.pth`。

## 模型評估與結果視覺化

```bash
python evaluate.py
```
- 終端機會顯示精確率、召回率、準確率。
- 預測結果圖片會儲存在 `predictions/` 目錄。

## 注意事項
- 若遇到 CUDA 記憶體不足，可將 `train.py` 及 `evaluate.py` 內的 `batch_size` 或 `num_workers` 調小。
- 若要重新訓練，請先刪除舊的 `checkpoints/faster_rcnn_model.pth`。

## 檔案結構
```
phase3/object_detection/
├── checkpoints/                # 儲存模型權重
├── data/                       # 資料集目錄
│   └── vehicles_images/        # 圖片與標註檔案
├── predictions/                # 預測結果圖片
├── dataset.py                  # 資料集定義
├── model.py                    # 模型與訓練邏輯
├── train.py                    # 訓練腳本
├── evaluate.py                 # 評估與視覺化腳本
├── requirements.txt            # 依賴套件
└── README.md                   # 使用說明
``` 