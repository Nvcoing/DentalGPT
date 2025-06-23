import requests
import time
from config import NGROK_URL

def build_prompt(prompt: str, rag_context: str = "") -> str:
    return (
        "<｜begin▁of▁sentence｜>"
        "<｜system｜>\n"
        f"""### Hướng dẫn: Bạn là DentalGPT — một trợ lý ảo thông minh chuyên hỗ trợ tư vấn và cung cấp kiến thức nha khoa cho bệnh nhân, sinh viên, và bác sĩ. 
        Bạn có khả năng truy xuất và sử dụng thông tin thu thập được từ Internet hoặc hệ thống cơ sở dữ liệu học thuật để phản hồi chính xác và cập nhật.
        Nếu có đoạn thông tin truy xuất (retrieved context), hãy ưu tiên sử dụng thông tin đó để trả lời.\n"""
        "<｜user｜>\n"
        f"### Câu hỏi:\n{prompt.strip()}\n\n"
        f"### Thông tin truy xuất được:\n{rag_context.strip()}\n"
        "<｜think｜>\n"
        "Hãy cùng diễn giải từng bước nào!🤔\n"
        "<reasoning_cot>\n"
        "# 🧠 Suy luận của DentalGPT\n"
        f"## 1️⃣ Mục tiêu 📌\nTrả lời đơn giản, đúng trọng tâm, ngắn gọn, dễ hiểu\n"
        f"## 2️⃣ Bước suy nghĩ ⚙️\nBước 1: Xác định đúng câu hỏi\nBước 2: Xác định câu trả lời\nBước 3: Xác định cách trình bày\n"
        f"## 3️⃣ Giải thích 📝\nGiải thích ngắn gọn\n"
        "</reasoning_cot>\n"
    )

def generate_response(prompt: str, rag_context:str, temperature=0.1, top_p=0.9, top_k=50,
                      repetition_penalty=1.0, do_sample=True, max_new_tokens=256):

    full_prompt = build_prompt(prompt, rag_context=rag_context)

    data = {
        "prompt": full_prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample
    }

    try:
        response = requests.post(NGROK_URL, json=data, stream=True)
        response.raise_for_status()
        time.sleep(0.5)
    except requests.exceptions.RequestException as e:
        yield f"Error during generation: {str(e)}"
        return

    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            yield chunk.decode("utf-8")
