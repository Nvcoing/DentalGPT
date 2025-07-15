import requests
import time
from config import NGROK_URL

def build_prompt(prompt: str) -> str:
    return (
        "<｜begin▁of▁sentence｜>"
        "<｜system｜>\n"
        "### Hướng dẫn: Hãy là một trợ lý ảo nha khoa DentalGPT - Hãy suy luận và phân biệt nên trình bày dạng bảng, biểu đồ, công thức. Nếu câu hỏi về bảng thì chuyên gia phải trình bày dạng bảng\n"
        "<｜user｜>\n"
        f"### Câu hỏi:\n{prompt.strip()}\n"
    )

def generate_response(prompt: str, temperature=0.1, top_p=0.9, top_k=50,
                      repetition_penalty=1.0, do_sample=True, max_new_tokens=1024):
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
