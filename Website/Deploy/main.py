from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from googlesearch import search
from bs4 import BeautifulSoup
import requests
import google.generativeai as genai

# Cấu hình Gemini 
genai.configure(api_key="AIzaSyApEktQbsw89BGSmTCRspL4xIm0UcBKo4Y")
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInput(BaseModel):
    message: str

# =============================
# 🔍 Retrieval: Google Search + crawl nội dung
# =============================
def retrieval(query, max_links=3):
    results = []
    for url in search(query, num_results=max_links):
        try:
            r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(r.text, "html.parser")
            # Lấy nội dung chính
            paragraphs = soup.find_all("p")
            content = "\n".join(p.get_text() for p in paragraphs)
            results.append({
                "url": url,
                "content": content  # Giới hạn tránh quá dài
            })
        except:
            continue
    return results

# =============================
# 🧠 Augmented: đưa vào Gemini để trả lời
# =============================
def augmented(query, documents):
    context = "\n\n".join(f"URL: {doc['url']}\n{doc['content']}" for doc in documents)
    prompt = f"""Dựa trên các nội dung sau từ các trang web, hãy trả lời câu hỏi bên dưới một cách ngắn gọn, súc tích:

{context}

Câu hỏi: {query}
"""
    response = model.generate_content(prompt)
    return response.text

# =============================
# 📩 API Endpoint
# =============================
@app.post("/chat")
async def chat(input: ChatInput):
    docs = retrieval(input.message)
    if not docs:
        return {"reply": "Không tìm thấy nội dung phù hợp."}
    try:
        summary = augmented(input.message, docs)
    except Exception as e:
        summary = "Lỗi xử lý Gemini: " + str(e)
    return {"reply": summary}
