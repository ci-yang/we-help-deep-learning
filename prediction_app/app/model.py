from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from gensim.models.doc2vec import Doc2Vec


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


def load_models():
    # 載入 Doc2Vec 模型
    doc2vec_path = Path(__file__).parent / "models" / "doc2vec_model.model"
    doc2vec_model = Doc2Vec.load(str(doc2vec_path))
    vector_size = doc2vec_model.vector_size

    # 載入分類器模型
    classifier_path = Path(__file__).parent / "models" / "classifier_model.pth"
    classifier_model = Classifier(input_dim=vector_size, hidden_dim=64, num_classes=9)
    classifier_model.load_state_dict(torch.load(classifier_path))
    classifier_model.eval()

    return doc2vec_model, classifier_model


def get_embedding(tokens: List[str], d2v_model: Any) -> List[float]:
    """將標題轉換為 embedding 向量"""
    vector_size = d2v_model.vector_size
    zero_vector = np.zeros(vector_size)

    # 只保留在詞彙表中的詞
    valid_words = [word for word in tokens if word in d2v_model.wv]

    if valid_words:
        # 如果有有效的詞，計算平均值
        embedding = np.mean([d2v_model.wv[word] for word in valid_words], axis=0)
    else:
        # 如果沒有有效的詞，使用零向量
        embedding = zero_vector

    return embedding.tolist()


def predict_title(
    title: str, doc2vec_model: Any, classifier_model: nn.Module
) -> Dict[str, Any]:
    """
    處理標題預測的函數

    Args:
        title (str): 使用者輸入的標題
        doc2vec_model (Any): Doc2Vec 模型
        classifier_model (nn.Module): 分類器模型

    Returns:
        Dict[str, Any]: 包含預測結果的字典
    """
    try:
        # 將標題分割為 tokens
        tokens = title.split()

        # 使用 Doc2Vec 將標題轉換為向量
        title_vector = get_embedding(tokens, doc2vec_model)

        # 將向量轉換為 PyTorch tensor
        title_tensor = torch.FloatTensor(title_vector).unsqueeze(0)

        # 使用分類器進行預測
        with torch.no_grad():
            output = classifier_model(title_tensor)
            predicted_class = int(torch.argmax(output, dim=1).item())

        # 類別映射
        class_mapping = {
            0: "Boy-Girl",
            1: "Lifeismoney",
            2: "Military",
            3: "Tech_Job",
            4: "baseball",
            5: "c_chat",
            6: "hatepolitics",
            7: "pc_shopping",
            8: "stock",
        }
        predicted_label = class_mapping[predicted_class]

        return {
            "status": "success",
            "predicted_board": predicted_label,
            "input_title": title,
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "input_title": title}
