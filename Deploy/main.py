from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import time
app = FastAPI()

# Cấu hình CORS để cho phép truy cập từ mọi nguồn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# URL của mô hình xử lý văn bản (cập nhật nếu cần)
NGROK_URL = " https://8d8a-34-125-84-96.ngrok-free.app/model/generate/"

@app.post("/DentalGPT/chatbot/")
async def generate(request: Request):
    req_json = await request.json()
    prompt = req_json.get("prompt")
    mode = req_json.get("mode", "normal")  # Mặc định là chế độ 'normal'
    max_new_tokens = req_json.get("max_new_tokens", 1024)
    temperature = req_json.get("temperature", 0.7)
    top_p = req_json.get("top_p", 0.9)
    top_k = req_json.get("top_k", 50)
    repetition_penalty = req_json.get("repetition_penalty", 1.0)
    do_sample = req_json.get("do_sample", True)
    num_samples = 1
    if not prompt:
        return JSONResponse(status_code=400, content={"error": "Missing 'prompt' in request"})

    # Tạo prompt dựa trên chế độ
    if mode == "reason":
        max_new_tokens = 1024  # Giới hạn số token cho chế độ 'reason'
        full_prompt = (
            "<｜begin▁of▁sentence｜>"
            "<｜system｜>\n"
            "### Hướng dẫn: Hãy là một trợ lý ảo nha khoa và SUY LUẬN để trả lời câu hỏi dưới đây:\n"
            "<｜user｜>\n"
            f"### Câu hỏi:\n{prompt.strip()}\n"
        )
    elif mode == "deep_reason":
        num_samples = 3
        max_new_tokens = 512
        full_prompt = (
            "<｜begin▁of▁sentence｜>"
            "<｜system｜>\n"
            "### Hướng dẫn: Hãy là một trợ lý ảo nha khoa và TRÌNH BÀY để trả lời câu hỏi dưới đây:\n"
            "<｜user｜>\n"
            f"### Câu hỏi:\n{prompt.strip()}\n"
        )
    else:
        max_new_tokens = 512  # Giới hạn số token cho chế độ 'normal'
        full_prompt = (
            "<｜begin▁of▁sentence｜>"
            "<｜system｜>\n"
            "### Hướng dẫn: Hãy là là một trợ lý ảo nha khoa và trả lời câu hỏi dưới đây:\n"
            "<｜user｜>\n"
            f"### Câu hỏi:\n{prompt.strip()}\n"
            "<｜think｜>\n"
            "Hãy cùng diễn giải từng bước nào!🤔\n"
            "<reasoning_cot>\n"
            "# 🧠 Suy luận của DentalGPT\n"
            f"## 1️⃣ Mục tiêu 📌\nTrả lời đơn giản, đúng trọng tâm, ngắn gọn, dễ hiểu\n"
            f"## 2️⃣ Bước suy nghĩ ⚙️\nBước 1: Xác định đúng câu hỏi\nBước 2: Xác định câu trả lời\nBước 3: Xác định cách trình bày\n"
            f"## 3️⃣ Giải thích 📝\nGiải thích ngắn gọn\n"
            "</reasoning_cot>\n"
        )
    
    data = {
        "prompt": full_prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample
    }
    deep_response = ""
    for _ in range(num_samples):
        deep_response+=full_prompt
        try:
            response = requests.post(NGROK_URL, json=data, stream=True)
            response.raise_for_status()
            deep_response+=f"\n{response}"
            time.sleep(0.5)  # tránh gửi quá nhanh gây lỗi server
        except requests.exceptions.RequestException as e:
            return JSONResponse(status_code=500, content={"error": f"Error during generation: {str(e)}"})

    def event_stream():
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                yield chunk.decode("utf-8")

    return StreamingResponse(event_stream(), media_type="text/markdown")
