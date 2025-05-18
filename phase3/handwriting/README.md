# 英文字母和數字識別 CNN 模型

這個專案實現了一個卷積神經網絡（CNN）模型，用於識別手寫的英文字母和數字。

## 資料集設置

在開始訓練模型之前，請先下載並設置資料集：

1. 從 Google Drive 下載資料集：
   - 資料集連結：[手寫英文字母和數字資料集](https://drive.google.com/file/d/1Uj4nROHC69BLKnb60qwq-DM_ClWbobgR/view)

2. 下載後，請將資料解壓縮到以下目錄結構：
   ```
   phase3/
   └── data/
       └── handwriting/
           ├── augmented_images/
           │   └── augmented_images1/  # 訓練資料圖片
           ├── handwritten-english-characters-and-digits/
           │   └── combined_folder/
           │       └── test/  # 測試資料圖片
           └── image_labels.csv  # 訓練資料標籤
   ```

## 使用說明

訓練模型：
```bash
python3 letter_digit_cnn.py
```

## 注意事項

- 請確保已安裝所需的 Python 套件（PyTorch、Pandas 等）
- 訓練好的模型會儲存在 `data/handwriting/best_cnn.pth` 