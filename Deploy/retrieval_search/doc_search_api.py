import os
import requests
import logging
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hàm lấy nội dung trang web
def get_page_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = "\n".join([p.get_text() for p in paragraphs])
        return content.strip()
    except requests.exceptions.RequestException as e:
        logger.warning(f"[LỖI] Không lấy được nội dung từ {url}: {e}")
        return ""

# Hàm chính: thực hiện tìm kiếm và trích xuất nội dung
def tool_search(query, api_key=API_KEY, cse_id=CSE_ID, num_results=1):
    if not api_key or not cse_id:
        logger.error("API_KEY hoặc CSE_ID không được tìm thấy trong biến môi trường.")
        return []

    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query, cx=cse_id, num=num_results).execute()

        results = []
        if 'items' in res:
            for item in res['items']:
                link = item.get('link')
                content = get_page_content(link)
                results.append({
                    'title': item.get('title'),
                    'link': link,
                    'snippet': item.get('snippet'),
                    'content': content
                })
        else:
            logger.info("Không tìm thấy kết quả phù hợp.")
        return results
    except Exception as e:
        logger.error(f"Lỗi trong tool_search: {e}")
        return []

# # Ví dụ gọi hàm
# if __name__ == "__main__":
#     query = "DFT technology JSC là công ty gì"
#     search_results = tool_search(query)

#     for i, item in enumerate(search_results):
#         print(f"{i+1}. 🏷️ {item['title']}")
#         print(f"   🔗 {item['link']}")
#         print(f"   ✍️  {item['snippet']}")
#         print(f"   📄 Nội dung:\n{item['content'][:800]}")  # In tối đa 800 ký tự
#         print("-" * 80)
