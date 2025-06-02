from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import requests
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hoặc chỉ định ["http://localhost:3000"] nếu dùng React, v.v.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
NGROK_URL = "https://1011-35-192-171-56.ngrok-free.app/model/generate/"

@app.post("/DentalGPT/chatbot/")
async def generate(request: Request):
    req_json = await request.json()
    prompt = req_json.get("prompt")
 # input nối
    max_new_tokens = req_json.get("max_new_tokens", 30)

    if not prompt:
        return JSONResponse(status_code=400, content={"error": "Missing 'prompt' in request"})

    # Nối thêm input nối vào prompt, bạn có thể tùy chỉnh cách nối (ví dụ thêm dấu cách)
    full_prompt = ("<｜begin▁of▁sentence｜>"
                "<｜system｜>\n"
                f"### Hướng dẫn: Hãy là là một trợ lý áo nha khoa và trả lời câu hỏi dưới đây:\n"
                "<｜user｜>\n"
                f"### Câu hỏi:\n {prompt.strip()}\n")

    data = {
        "prompt": full_prompt,
        "max_new_tokens": max_new_tokens
    }

    # Gọi API bên ngoài với stream=True để stream dữ liệu về
    external_response = requests.post(NGROK_URL, json=data, stream=True)

    def event_stream():
        for chunk in external_response.iter_content(chunk_size=None):
            if chunk:
                yield chunk.decode("utf-8")

    return StreamingResponse(event_stream(), media_type="text/markdown")
