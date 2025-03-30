from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


@dataclass
class Document:
    words: List[str]
    tag: str


@dataclass
class DocumentProcessor:
    def load_raw_documents(self, file_path: str) -> List[Document]:
        df = pd.read_csv(file_path)
        documents = [
            Document(words=row["tokens"].split(), tag=str(idx))
            for idx, row in df.iterrows()
        ]
        return documents


@dataclass
class Doc2VecModel:
    vector_size: int = 200
    window_size: int = 2
    min_count: int = 1
    workers: int = 4
    epochs: int = 500
    model: Doc2Vec = field(init=False)

    def __post_init__(self):
        self.model = Doc2Vec(
            vector_size=self.vector_size,
            window=self.window_size,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            dm=0,
        )

    def train(self, documents: List[Document]) -> None:
        tagged_documents = [
            TaggedDocument(words=doc.words, tags=[doc.tag]) for doc in documents
        ]
        self.model.build_vocab(tagged_documents)
        self.model.train(
            tagged_documents,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs,
        )

    def evaluate(self, documents: List[Document]) -> float:
        total_docs = len(documents)
        self_similarity_count = 0
        second_similarity_count = 0

        for idx, doc in enumerate(documents):
            inferred_vector = self.model.infer_vector(doc.words)
            sims = self.model.dv.most_similar([inferred_vector], topn=2)
            if int(sims[0][0]) == idx:
                self_similarity_count += 1

            if idx in [int(sims[0][0]), int(sims[1][0])]:
                second_similarity_count += 1

        self_similarity = self_similarity_count / total_docs
        second_similarity = second_similarity_count / total_docs

        print(f"✅ Self-Similarity: {self_similarity:.4f}")
        print(f"✅ Second Self-Similarity: {second_similarity:.4f}")

        return second_similarity

    def save(self, file_path: str) -> None:
        self.model.save(file_path)
        print(f"💾 模型已儲存至 {file_path}")


def execute_doc2vec_training():
    processor = DocumentProcessor()
    print("📥 載入原始標題格式資料中...")
    documents = processor.load_raw_documents("data/tokenized_data.csv")
    print(f"📄 共載入 {len(documents)} 筆標題資料")

    print("🚀 開始訓練 Doc2Vec 模型...")
    model = Doc2VecModel()
    model.train(documents)

    print("📊 進行模型評估...")
    second_similarity = model.evaluate(documents)

    if second_similarity >= 0.8:
        model.save("doc2vec_original_optimized.model")
    else:
        print("❌ Second Self-Similarity 未達 80%，模型未儲存。")


if __name__ == "__main__":
    execute_doc2vec_training()
