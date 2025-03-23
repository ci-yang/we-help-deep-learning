import re

import pandas as pd
from ckip_transformers.nlp import CkipWordSegmenter

# 初始化 CKIP 斷詞器
ws_driver = CkipWordSegmenter(model="bert-base", device=-1)  # 使用 CPU

# 介係詞與連接詞清單
stopwords = [
    "的",
    "了",
    "在",
    "是",
    "和",
    "及",
    "與",
    "或",
    "而",
    "並",
    "且",
    "但",
    "因為",
    "所以",
    "如果",
    "那麼",
    "然而",
    "雖然",
    "但是",
    "即使",
    "除了",
    "之後",
    "之前",
    "於",
    "於是",
    "因此",
    "等等",
    "對於",
    "通過",
    "由於",
    "至於",
    "關於",
    "由",
    "自",
    "從",
    "往",
    "向",
    "跟",
    "和",
    "與",
    "對",
    "給",
    "比",
    "像",
]


def is_valid_token(token):
    """判斷是否為有效詞"""
    return token not in stopwords and re.match(r"^[\u4e00-\u9fa5a-zA-Z0-9]+$", token)


def tokenize_title(title):
    """使用 CKIP 進行斷詞並過濾"""
    ws_result = ws_driver([title])
    tokens = [token for token in ws_result[0] if is_valid_token(token)]
    return " ".join(tokens)  # 用空白連接


def process_and_save(
    input_file="data/cleaned_data.csv", output_file="data/tokenized_data.csv"
):
    """處理資料並存檔"""
    df = pd.read_csv(input_file)
    df.drop_duplicates(subset=["Title"], inplace=True)

    tokenized_rows = []

    for _, row in df.iterrows():
        label = row["Board"]
        title = row["Title"]
        tokens = tokenize_title(title)
        if tokens:  # 過濾掉空白結果
            tokenized_rows.append([label, tokens])

    tokenized_df = pd.DataFrame(tokenized_rows, columns=["board", "tokens"])
    tokenized_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✅ 斷詞後的資料已儲存至 {output_file}")


if __name__ == "__main__":
    process_and_save()
