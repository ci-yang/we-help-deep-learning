from __future__ import annotations

import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException


@dataclass
class PTTCrawler:
    boards: List[str]
    base_url: str = "https://www.ptt.cc/bbs/{}/index.html"
    headers: dict = field(default_factory=lambda: {"User-Agent": "Mozilla/5.0"})
    max_titles: int = 10000
    max_pages: int = 2000
    max_workers: int = 4

    def fetch_titles(self, board: str) -> List[Tuple[str, str]]:
        titles = []
        url = self.base_url.format(board)
        count = 0
        page = 0

        print(f"\n開始爬取 {board} 看板...")
        while url and count < self.max_titles and page < self.max_pages:
            try:
                if page % 10 == 0:
                    print(f"{board} 看板進度: 第 {page} 頁, 已取得 {count} 篇文章")
                
                response = requests.get(
                    url, 
                    headers=self.headers, 
                    cookies={"over18": "1"},
                    timeout=30
                )
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")

                page_titles = []
                for title_tag in soup.select(".title a"):
                    try:
                        title = title_tag.text.strip()
                        if title:  # 確保標題不為空
                            page_titles.append((board, title))
                            count += 1
                            if count >= self.max_titles:
                                break
                    except AttributeError as e:
                        print(f"⚠️ {board} 看板解析標題時發生錯誤: {e}")
                        continue
                
                titles.extend(page_titles)
                print(f"{board} 看板: 第 {page} 頁完成, 本頁 {len(page_titles)} 篇文章")

                try:
                    prev_link = soup.select_one(".btn-group-paging a:nth-child(2)")
                    prev_link_href = prev_link["href"] if prev_link else None
                    url = f"https://www.ptt.cc{prev_link_href}" if prev_link_href else None
                except (AttributeError, KeyError) as e:
                    print(f"⚠️ {board} 看板解析下一頁連結時發生錯誤: {e}")
                    break

                page += 1
                time.sleep(1)

            except RequestException as e:
                print(f"❌ {board} 看板存取錯誤: {e}")
                time.sleep(5)  # 發生錯誤時等待較長時間
                continue
            except Exception as e:
                print(f"❌ {board} 看板發生未預期的錯誤: {e}")
                break

        print(f"✅ {board} 看板爬取完成, 總共 {count} 篇文章")
        return titles

    def fetch_all_titles(self) -> List[Tuple[str, str]]:
        all_titles = []
        lock = threading.Lock()
        completed_boards = 0

        def fetch_board(board: str) -> None:
            nonlocal completed_boards
            try:
                titles = self.fetch_titles(board)
                with lock:
                    all_titles.extend(titles)
                    completed_boards += 1
                    print(f"\n進度: {completed_boards}/{len(self.boards)} 個看板完成")
                    print(f"目前共收集 {len(all_titles)} 篇文章")
            except Exception as e:
                print(f"❌ {board} 看板處理失敗: {e}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(fetch_board, self.boards)

        return all_titles


@dataclass
class DataProcessor:
    def save_raw_data(
        self, data: List[Tuple[str, str]], filename: str = "raw_data.csv"
    ) -> None:
        try:
            df = pd.DataFrame(data, columns=["Board", "Title"])
            df.to_csv(filename, index=False, encoding="utf-8-sig")
            print(f"✅ 已儲存原始資料到 {filename}")
        except Exception as e:
            print(f"❌ 儲存原始資料時發生錯誤: {e}")

    def clean_data(
        self, input_file: str = "raw_data.csv", output_file: str = "cleaned_data.csv"
    ) -> None:
        try:
            df = pd.read_csv(input_file)
            df["Title"] = df["Title"].apply(
                lambda x: re.sub(r"^(Re:|Fw:)\s*", "", x, flags=re.IGNORECASE)
            )
            df["Title"] = df["Title"].str.strip().str.lower()
            df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"✅ 已儲存清理後的資料到 {output_file}")
        except Exception as e:
            print(f"❌ 清理資料時發生錯誤: {e}")


def execute_crawling():
    try:
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

        if all_titles:  # 確保有爬到資料才進行處理
            processor.save_raw_data(all_titles)
            processor.clean_data()
        else:
            print("❌ 沒有爬取到任何資料")

    except Exception as e:
        print(f"❌ 執行過程中發生未預期的錯誤: {e}")


if __name__ == "__main__":
    execute_crawling()
