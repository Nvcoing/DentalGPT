import requests
import time
from config import NGROK_URL

def build_prompt(prompt: str) -> str:
    return (
        "<｜begin▁of▁sentence｜>"
        "<｜system｜>\n"
        f"""### Hướng dẫn: Bạn là DentalGPT — một trợ lý ảo về y tế.\n"""
        "<｜user｜>\n"
        f"### Câu hỏi:\n{prompt.strip()}\n\n"
        "<｜think｜>\n"
        "Hãy cùng diễn giải từng bước nào!🤔\n"
        "<reasoning_cot>\n"
        "# 🧠 Suy luận của DentalGPT\n"
        f"## 1️⃣ Mục tiêu 📌\nTrả lời đơn giản, đúng trọng tâm, ngắn gọn, dễ hiểu, đúng chuyên môn\n"
        f"## 2️⃣ Bước suy nghĩ ⚙️\nBước 1: Xác định trọng tâm câu hỏi\nBước 2: Trích xuất thông tin y khoa liên quan \nBước 3: Diễn giải theo cách đơn giản, dễ tiếp thu\n"
        f"## 3️⃣ Giải thích 📝\nTrình bày nguyên nhân, tác động và hướng xử lý một cách ngắn gọn, dễ hiểu, tránh thuật ngữ phức tạp.\n"
        "</reasoning_cot>\n"
    )

def generate_response(prompt: str, temperature=0.1, top_p=0.9, top_k=50,
                      repetition_penalty=1.0, do_sample=True, max_new_tokens=256):

    full_prompt = build_prompt(prompt)

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
