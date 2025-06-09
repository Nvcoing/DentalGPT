import requests
import time
from config import NGROK_URL
from gemini_tool.call_gemini import call_gemini

# Xây dựng prompt đầu vào cơ bản cho LLM
def build_prompt(prompt: str) -> str:
    return (
        "<｜begin▁of▁sentence｜>"
        "<｜system｜>\n"
        "### Hướng dẫn: Hãy là một trợ lý ảo nha khoa và TRÌNH BÀY để trả lời câu hỏi dưới đây:\n"
        "<｜user｜>\n"
        f"### Câu hỏi:\n{prompt.strip()}\n"
    )

# Gửi prompt đến server LLM qua API
def send_request(prompt: str, generation_params: dict):
    try:
        response = requests.post(NGROK_URL, json={"prompt": prompt, **generation_params}, stream=True)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        return f"Error during generation: {str(e)}"

# Tạo prompt yêu cầu Gemini phản biện và gợi ý cải tiến
def build_refining_prompt(question_prompt: str, raw_answer: str) -> str:
    return f"""Bạn là một chuyên gia tạo prompt và đào tạo mô hình ngôn ngữ lớn (LLM). Nhiệm vụ của bạn là giúp cải thiện khả năng suy luận, hiểu sâu và học hỏi của mô hình trong lĩnh vực **nha khoa**.

Tôi sẽ cung cấp:
- Một **câu hỏi về nha khoa**.
- Một **câu trả lời hiện tại** (có thể đúng, sai, hoặc chưa đầy đủ).

Bạn cần thực hiện các bước sau:

1. **Phân tích câu trả lời hiện tại**:
   - Nó đúng, sai, hay thiếu sót?
   - Có vấn đề nào về suy luận, logic, độ đầy đủ, hay độ chính xác?

2. **Gợi ý sửa câu trả lời**:
   - Viết lại câu trả lời sao cho **hoàn chỉnh, logic và chuyên sâu hơn**, phù hợp với kiến thức nha khoa hiện đại.

3. **Tạo một prompt mới tối ưu cho LLM**:
   - Prompt này nên giúp mô hình trả lời sâu sắc hơn trong tương lai.
   - Khuyến khích mô hình sử dụng kiến thức chuyên môn, giải thích nguyên nhân, hậu quả, và ví dụ minh họa.

4. **Gợi ý kiến thức cần ôn tập hoặc tìm hiểu thêm**:
   - Nếu câu trả lời sai hoặc thiếu, hãy liệt kê khái niệm nên ôn lại (ví dụ: viêm lợi, cấu trúc răng hàm, kỹ thuật nhổ răng…).

5. **Gợi ý một vài câu hỏi mở rộng liên quan**:
   - Ví dụ: “Nếu răng khôn mọc lệch không gây đau, có cần nhổ không?”, “Các biến chứng nếu không nhổ răng khôn là gì?” v.v.

---

**Câu hỏi:** {question_prompt.strip()}  
**Câu trả lời hiện tại:** {raw_answer.strip()}

Hãy thực hiện đầy đủ 5 bước ở trên cho ví dụ này.
"""

# Hàm chính xử lý sinh câu trả lời, tinh chỉnh và stream output
def generate_response(prompt: str,
                      temperature=0.1, top_p=0.9, top_k=50,
                      repetition_penalty=1.0, do_sample=True,
                      max_new_tokens=1024):
    
    generation_params = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample
    }

    # Bước 1: Tạo prompt cơ bản và lấy câu trả lời đầu tiên
    base_prompt = build_prompt(prompt)
    initial_response = send_request(base_prompt, generation_params)
    
    if isinstance(initial_response, str):
        yield initial_response
        return
    
    raw_answer = initial_response.text.strip()

    # Bước 2: Gửi cho Gemini để phản biện và tạo prompt tốt hơn
    refining_prompt = build_refining_prompt(base_prompt, raw_answer)
    improved_instruction = call_gemini(refining_prompt, model_name="models/gemini-1.5-flash-latest")

    # Bước 3: Tạo prompt mới từ phản hồi của Gemini
    refined_prompt = (
        "<｜begin▁of▁sentence｜>"
        "<｜system｜>\n"
        f"### Hướng dẫn: \n{improved_instruction.strip()}\n"
        "<｜user｜>\n"
        f"### Câu hỏi:\n{prompt.strip()}\n"
    )

    # Bước 4: Gửi prompt cải tiến và stream kết quả
    final_response = send_request(refined_prompt, generation_params)
    if isinstance(final_response, str):
        yield final_response
        return

    time.sleep(0.5)  # Chờ tránh quá tải

    for chunk in final_response.iter_content(chunk_size=None):
        if chunk:
            yield chunk.decode("utf-8")
