from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup


@dataclass
class PTTCrawler:
    boards: List[str]
    base_url: str = "https://www.ptt.cc/bbs/{}/index.html"
    headers: dict = field(default_factory=lambda: {"User-Agent": "Mozilla/5.0"})
    max_titles: int = 1000
    max_pages: int = 2000
    max_workers: int = 8

    def fetch_titles(self, board: str) -> List[Tuple[str, str]]:
        titles = []
        url = self.base_url.format(board)
        count = 0
        page = 0

        print(f"\n開始爬取 {board} 看板...")
        while url and count < self.max_titles and page < self.max_pages:
            if page % 10 == 0:
                print(f"{board} 看板進度: 第 {page} 頁, 已取得 {count} 篇文章")

            response = requests.get(url, headers=self.headers, cookies={"over18": "1"})
            if response.status_code != 200:
                print(f"❌ 無法訪問 {board}")
                break

            soup = BeautifulSoup(response.text, "html.parser")

            page_titles = []
            for title_tag in soup.select(".title a"):
                title = title_tag.text.strip()
                page_titles.append((board, title))
                count += 1
                if count >= self.max_titles:
                    break

            titles.extend(page_titles)
            print(f"{board} 看板: 第 {page} 頁完成, 本頁 {len(page_titles)} 篇文章")

            prev_link = soup.select_one(".btn-group-paging a:nth-child(2)")
            prev_link_href = prev_link["href"] if prev_link else None
            url = f"https://www.ptt.cc{prev_link_href}" if prev_link_href else None
            page += 1

            time.sleep(1)

        print(f"✅ {board} 看板爬取完成, 總共 {count} 篇文章")
        return titles

    def fetch_all_titles(self) -> List[Tuple[str, str]]:
        all_titles = []
        lock = threading.Lock()
        completed_boards = 0

        def fetch_board(board: str) -> None:
            nonlocal completed_boards
            titles = self.fetch_titles(board)
            with lock:
                all_titles.extend(titles)
                completed_boards += 1
                print(f"\n進度: {completed_boards}/{len(self.boards)} 個看板完成")
                print(f"目前共收集 {len(all_titles)} 篇文章")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(fetch_board, self.boards)

        return all_titles


@dataclass
class DataProcessor:
    def save_raw_data(
        self, data: List[Tuple[str, str]], filename: str = "raw_data.csv"
    ) -> None:
        df = pd.DataFrame(data, columns=["Board", "Title"])
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"✅ 已儲存原始資料到 {filename}")

    def clean_data(
        self, input_file: str = "raw_data.csv", output_file: str = "cleaned_data.csv"
    ) -> None:
        df = pd.read_csv(input_file)
        df["Title"] = df["Title"].apply(
            lambda x: re.sub(r"^(Re:|Fw:)\s*", "", x, flags=re.IGNORECASE)
        )
        df["Title"] = df["Title"].str.strip().str.lower()
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"✅ 已儲存清理後的資料到 {output_file}")


def execute_crawling():
    boards = [
        "baseball",
        "Boy-Girl",
        "c_chat",
        "hatepolitics",
        "Lifeismoney",
        "Military",
        "pc_shopping",
        "stock",
        "Tech_Job",
    ]

    crawler = PTTCrawler(boards=boards)
    processor = DataProcessor()

    print(f"開始爬取 {len(boards)} 個看板...")
    all_titles = crawler.fetch_all_titles()
    print(f"\n爬取完成，共 {len(all_titles)} 篇文章")

    processor.save_raw_data(all_titles)
    processor.clean_data()


if __name__ == "__main__":
    execute_crawling()
