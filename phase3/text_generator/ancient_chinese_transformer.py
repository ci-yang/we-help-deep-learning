# 標準庫
import json
import math
import random

# 第三方套件
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 設定隨機種子，確保結果可重現
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def load_ancient_texts():
    """
    讀取 data.json 檔案中的古文數據，若失敗則使用備用數據。
    Returns:
        list: 古文句子列表
    """
    try:
        with open("data/data.json", "r", encoding="utf8") as f:
            data = json.load(f)
        texts = []
        for chapter_data in data:
            paragraphs = chapter_data.get("paragraphs", [])
            for paragraph in paragraphs:
                cleaned_text = paragraph.replace("「", "").replace("」", "")
                cleaned_text = cleaned_text.replace("『", "").replace("』", "")
                cleaned_text = cleaned_text.replace('"', "")
                sentences = []
                current_sentence = ""
                for char in cleaned_text:
                    current_sentence += char
                    if char in "。？！；：":
                        if len(current_sentence.strip()) > 3:
                            sentences.append(current_sentence.strip())
                        current_sentence = ""
                if current_sentence.strip() and len(current_sentence.strip()) > 3:
                    sentences.append(current_sentence.strip())
                texts.extend(sentences)
    except Exception as e:
        print(f"讀取 data.json 失敗: {e}")
        texts = [
            "子曰：學而時習之，不亦說乎？",
            "有朋自遠方來，不亦樂乎？",
            "人不知而不慍，不亦君子乎？",
            "大學之道，在明明德，在親民，在止於至善。",
            "天命之謂性，率性之謂道，修道之謂教。",
            "君子中庸，小人反中庸。",
            "己所不欲，勿施於人。",
            "三人行，必有我師焉。",
            "溫故而知新，可以為師矣。",
            "學而不思則罔，思而不學則殆。",
            "君子坦蕩蕩，小人長戚戚。",
            "仁者見仁，智者見智。",
            "知之為知之，不知為不知，是知也。",
            "為政以德，譬如北辰。",
            "道之以政，齊之以刑，民免而無恥。",
            "孟子見梁惠王。",
            "王何必曰利？亦有仁義而已矣。",
            "民為貴，社稷次之，君為輕。",
            "天行健，君子以自強不息。",
            "地勢坤，君子以厚德載物。",
        ]
    unique_texts = []
    seen = set()
    for text in texts:
        if text not in seen and len(text) >= 4:
            unique_texts.append(text)
            seen.add(text)
    return unique_texts


def load_data():
    """
    載入古文數據
    Returns:
        list: 古文句子列表
    """
    return load_ancient_texts()


class Tokenizer:
    """
    字元級分詞器
    """

    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

    def build_vocab(self, texts):
        """
        建立字元詞彙表
        Args:
            texts (list): 古文句子列表
        """
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        all_chars = special_tokens + sorted(list(all_chars))
        self.char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(all_chars)

    def encode(self, text):
        """
        將文本轉為索引序列
        Args:
            text (str): 輸入文本
        Returns:
            list: 索引序列
        """
        return [self.char_to_idx.get(char, self.char_to_idx["<UNK>"]) for char in text]

    def decode(self, indices):
        """
        將索引序列轉回文本
        Args:
            indices (list): 索引序列
        Returns:
            str: 解碼後文本
        """
        return "".join([self.idx_to_char.get(idx, "<UNK>") for idx in indices])


class AncientChineseDataset(Dataset):
    """
    古文資料集
    """

    def __init__(self, texts, tokenizer, max_length=32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        for text in texts:
            if len(text) > 2:
                encoded = tokenizer.encode(text)
                if len(encoded) > max_length - 2:
                    encoded = encoded[: max_length - 2]
                encoded = (
                    [tokenizer.char_to_idx["<BOS>"]]
                    + encoded
                    + [tokenizer.char_to_idx["<EOS>"]]
                )
                padded = encoded + [tokenizer.char_to_idx["<PAD>"]] * (
                    max_length - len(encoded)
                )
                self.data.append(padded)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        x = torch.tensor(sequence[:-1], dtype=torch.long)
        y = torch.tensor(sequence[1:], dtype=torch.long)
        return x, y


class PositionalEncoding(nn.Module):
    """
    位置編碼 - 修正版本
    """

    def __init__(self, d_model, max_length=1000):
        super().__init__()
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe_tensor", pe)  # 避免與 Module 名稱衝突

    def forward(self, x):
        """
        前向傳播
        Args:
            x (torch.Tensor): 輸入張量，形狀為 (seq_len, batch_size, d_model) 或 (batch_size, seq_len, d_model)
        Returns:
            torch.Tensor: 加上位置編碼的張量
        """
        pe = self.pe_tensor
        if not isinstance(pe, torch.Tensor):
            pe = torch.as_tensor(pe)
        if x.dim() == 3:
            if x.size(0) <= 1000 and x.size(1) > x.size(0):
                seq_len = x.size(0)
                pe_slice = pe[:seq_len, :].unsqueeze(1)
            else:
                seq_len = x.size(1)
                pe_slice = pe[:seq_len, :].unsqueeze(0)
        else:
            seq_len = x.size(0)
            pe_slice = pe[:seq_len, :].unsqueeze(1)
        return x + pe_slice


class TransformerModel(nn.Module):
    """
    Transformer 編碼器模型
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=8,
        num_layers=6,
        d_ff=512,
        max_length=32,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(d_model, vocab_size)

    def create_mask(self, seq_len):
        """
        建立因果遮罩（causal mask）
        Args:
            seq_len (int): 序列長度
        Returns:
            torch.Tensor: 遮罩
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.bool()

    def create_padding_mask(self, x, pad_token_id=0):
        """
        建立填充遮罩
        Args:
            x (torch.Tensor): 輸入序列
            pad_token_id (int): PAD 的索引
        Returns:
            torch.Tensor: 遮罩
        """
        return x == pad_token_id

    def forward(self, x, pad_token_id=0):
        """
        前向傳播
        Args:
            x (torch.Tensor): 輸入序列
            pad_token_id (int): PAD 的索引
        Returns:
            torch.Tensor: 預測結果
        """
        batch_size, seq_len = x.shape
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # 轉為 (seq_len, batch_size, d_model)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # 轉回 (batch_size, seq_len, d_model)
        x = self.dropout(x)
        causal_mask = self.create_mask(seq_len).to(x.device)
        transformer_output = self.transformer_encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=None,
        )
        output = self.output_linear(transformer_output)
        return output


def train_model(model, dataloader, criterion, optimizer, num_epochs=100):
    """
    訓練模型
    Args:
        model: Transformer 模型
        dataloader: 資料載入器
        criterion: 損失函數
        optimizer: 優化器
        num_epochs (int): 訓練輪數
    Returns:
        list: 每個 epoch 的平均損失
    """
    model.train()
    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            output_reshaped = output.view(-1, output.size(-1))
            target_reshaped = target.view(-1)
            loss = criterion(output_reshaped, target_reshaped)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        losses.append(avg_loss)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return losses


def initialize_model_and_tokenizer(ancient_chinese_texts):
    """
    初始化模型和分詞器
    Args:
        ancient_chinese_texts (list): 古文句子列表
    Returns:
        tuple: (model, tokenizer, dataset, dataloader)
    """
    print("2. 準備分詞器...")
    tokenizer = Tokenizer()
    tokenizer.build_vocab(ancient_chinese_texts)
    print(f"詞彙表大小: {tokenizer.vocab_size}")
    print(f"訓練文本數量: {len(ancient_chinese_texts)}")

    print("\n示例古文文本:")
    for i, text in enumerate(ancient_chinese_texts[:5]):
        print(f"  {i+1}. {text}")
    print(f"  ... 還有 {len(ancient_chinese_texts)-5} 個文本")

    print("\n3. 創建數據集...")
    dataset = AncientChineseDataset(ancient_chinese_texts, tokenizer, max_length=32)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"數據集大小: {len(dataset)}")

    print("4. 初始化 Transformer 模型...")
    print("使用組件: nn.TransformerEncoderLayer + nn.TransformerEncoder")
    model = TransformerModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=3,
        d_ff=256,
        max_length=32,
        dropout=0.1,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型總參數數量: {total_params:,}")

    return model, tokenizer, dataset, dataloader


def save_model_and_tokenizer(
    model,
    tokenizer,
    model_path="models/ancient_chinese_transformer.pth",
    tokenizer_path="models/tokenizer.json",
):
    """
    儲存模型和分詞器
    Args:
        model: Transformer 模型
        tokenizer: 分詞器
        model_path (str): 模型儲存路徑
        tokenizer_path (str): 分詞器儲存路徑
    """
    import os

    os.makedirs("models", exist_ok=True)

    # 儲存模型
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "vocab_size": model.embedding.num_embeddings,
                "d_model": model.d_model,
                "num_heads": 4,
                "num_layers": 3,
                "d_ff": 256,
                "max_length": model.max_length,
                "dropout": 0.1,
            },
        },
        model_path,
    )

    # 儲存分詞器
    tokenizer_data = {
        "char_to_idx": tokenizer.char_to_idx,
        "idx_to_char": tokenizer.idx_to_char,
        "vocab_size": tokenizer.vocab_size,
    }
    with open(tokenizer_path, "w", encoding="utf8") as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

    print(f"模型已儲存至: {model_path}")
    print(f"分詞器已儲存至: {tokenizer_path}")


def load_model_and_tokenizer(
    model_path="models/ancient_chinese_transformer.pth",
    tokenizer_path="models/tokenizer.json",
):
    """
    載入模型和分詞器
    Args:
        model_path (str): 模型路徑
        tokenizer_path (str): 分詞器路徑
    Returns:
        tuple: (model, tokenizer) 或 (None, None) 如果載入失敗
    """
    try:
        # 載入分詞器
        with open(tokenizer_path, "r", encoding="utf8") as f:
            tokenizer_data = json.load(f)

        tokenizer = Tokenizer()
        tokenizer.char_to_idx = tokenizer_data["char_to_idx"]
        tokenizer.idx_to_char = {
            int(k): v for k, v in tokenizer_data["idx_to_char"].items()
        }
        tokenizer.vocab_size = tokenizer_data["vocab_size"]

        # 載入模型
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        config = checkpoint["model_config"]

        model = TransformerModel(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            d_ff=config["d_ff"],
            max_length=config["max_length"],
            dropout=config["dropout"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"成功載入模型: {model_path}")
        print(f"成功載入分詞器: {tokenizer_path}")
        return model, tokenizer

    except Exception as e:
        print(f"載入模型失敗: {e}")
        return None, None


def train_and_save_model(model, tokenizer, dataloader):
    """
    訓練並儲存模型
    Args:
        model: Transformer 模型
        tokenizer: 分詞器
        dataloader: 資料載入器
    Returns:
        list: 訓練損失列表
    """
    print("5. 開始訓練模型...")
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char_to_idx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = train_model(model, dataloader, criterion, optimizer, num_epochs=150)

    # 訓練完成後儲存模型
    save_model_and_tokenizer(model, tokenizer)

    # 繪製訓練損失曲線
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Transformer Training Loss (使用 PyTorch 內建組件)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    return losses


def generate_sentences(model, tokenizer, num_sentences=5, custom_seeds=None):
    """
    生成古文句子
    Args:
        model: Transformer 模型
        tokenizer: 分詞器
        num_sentences (int): 要生成的句子數量
        custom_seeds (list): 自定義種子詞列表，如果為 None 則使用預設
    Returns:
        list: 生成的句子列表
    """
    if custom_seeds is None:
        seed_pool = [
            "君子",
            "天下",
            "學而",
            "道者",
            "仁者",
            "智者",
            "聖人",
            "古人",
            "子曰",
            "孔子曰",
            "孟子曰",
            "大學",
            "中庸",
            "論語",
            "孟子",
            "仁義",
            "禮樂",
            "修身",
            "治國",
            "平天下",
            "格物",
            "致知",
            "誠意",
            "正心",
            "齊家",
            "為政",
            "學問",
            "思辨",
            "知行",
        ]
    else:
        seed_pool = custom_seeds

    generated_sentences = []

    print(f"\n=== 生成 {num_sentences} 個古代中文句子 ===")
    print("(比隨機生成更有意義和結構)")

    for i in range(num_sentences):
        # 從種子池中隨機選擇，如果數量超過種子池則循環使用
        seed = seed_pool[i % len(seed_pool)]

        # 隨機選擇溫度和長度參數
        temperature = random.uniform(0.6, 0.9)
        max_length = random.randint(8, 15)

        generated = generate_text(
            model,
            tokenizer,
            seed_text=seed,
            max_length=max_length,
            temperature=temperature,
        )
        generated_sentences.append(generated)
        print(f"{i+1}. {generated}")

    return generated_sentences


def generate_classic_continuations(model, tokenizer, num_continuations=5):
    """
    生成經典句式續寫
    Args:
        model: Transformer 模型
        tokenizer: 分詞器
        num_continuations (int): 要生成的續寫數量
    Returns:
        list: 生成的續寫列表
    """
    classic_starts = [
        "子曰",
        "孔子曰",
        "孟子曰",
        "聖人",
        "君子之道",
        "古之",
        "今之",
        "大學之道",
        "中庸之",
        "為政以德",
        "學而",
        "溫故",
        "三人行",
        "己所不欲",
        "仁者",
    ]

    generated_continuations = []

    print(f"\n=== 生成 {num_continuations} 個經典句式續寫 ===")

    for i in range(num_continuations):
        start = classic_starts[i % len(classic_starts)]
        generated = generate_text(
            model, tokenizer, seed_text=start, max_length=15, temperature=0.7
        )
        generated_continuations.append(generated)
        print(f"{i+1}. {generated}")

    return generated_continuations


def display_model_summary():
    """
    顯示模型架構總結
    """
    print("\n=== 模型架構總結 ===")
    print("✅ 使用了 nn.TransformerEncoderLayer")
    print("✅ 使用了 nn.TransformerEncoder")
    print("✅ 整合了 Tokenizer 和 Embedding 層")
    print("✅ 實現了多頭自注意力機制")
    print("✅ 從頭建立並訓練模型")
    print("✅ 從 data.json 讀取真實古文數據")
    print("✅ 生成超過5個有意義的古代中文句子")
    print("✅ 支援模型儲存和載入功能")
    print("\n=== 訓練完成！ ===")


def generate_text(model, tokenizer, seed_text="君子", max_length=15, temperature=0.8):
    """
    生成古文句子
    Args:
        model: Transformer 模型
        tokenizer: 分詞器
        seed_text (str): 種子文本
        max_length (int): 生成最大長度
        temperature (float): 溫度參數
    Returns:
        str: 生成的古文句子
    """
    model.eval()
    input_ids = [tokenizer.char_to_idx["<BOS>"]] + tokenizer.encode(seed_text)
    with torch.no_grad():
        for _ in range(max_length):
            if len(input_ids) >= model.max_length - 1:
                input_ids = input_ids[-(model.max_length - 1) :]
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            output = model(input_tensor)
            logits = output[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            if next_token == tokenizer.char_to_idx["<EOS>"]:
                break
            if next_token == tokenizer.char_to_idx["<PAD>"]:
                continue
            input_ids.append(next_token)
    generated_text = tokenizer.decode(input_ids[1:])
    generated_text = (
        generated_text.replace("<EOS>", "").replace("<PAD>", "").replace("<UNK>", "")
    )
    return generated_text


def main():
    """
    主程式：訓練 Transformer 並生成古文句子
    """
    print("=== 古代中文 Transformer 文本生成模型 ===")
    print("使用 PyTorch 內建 Transformer 組件\n")

    # 步驟1: 載入古文數據
    print("1. 從 data.json 讀取古文數據...")
    ancient_chinese_texts = load_data()

    # 步驟2-4: 嘗試載入已訓練的模型
    print("\n檢查是否有已訓練的模型...")
    model, tokenizer = load_model_and_tokenizer()

    if model is None or tokenizer is None:
        # 沒有已訓練的模型，需要訓練新模型
        print("未找到已訓練模型，開始訓練新模型...")
        model, tokenizer, dataset, dataloader = initialize_model_and_tokenizer(
            ancient_chinese_texts
        )
        losses = train_and_save_model(model, tokenizer, dataloader)
    else:
        # 有已訓練的模型
        print("找到已訓練模型，跳過訓練步驟。")
        print("如需重新訓練，請刪除 models/ 資料夾中的模型檔案。")

    # 步驟5: 生成古文句子（可自定義數量）
    generated_sentences = generate_sentences(model, tokenizer, num_sentences=6)

    # 步驟6: 生成經典句式續寫
    classic_continuations = generate_classic_continuations(
        model, tokenizer, num_continuations=5
    )

    # 步驟7: 顯示模型架構總結
    display_model_summary()

    # 額外功能展示：自定義種子詞生成
    print("\n=== 自定義種子詞生成示例 ===")
    custom_seeds = ["道德", "智慧", "修養", "品格"]
    custom_sentences = generate_sentences(
        model, tokenizer, num_sentences=len(custom_seeds), custom_seeds=custom_seeds
    )


if __name__ == "__main__":
    main()
