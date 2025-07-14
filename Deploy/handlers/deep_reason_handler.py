import requests
import time
from config import NGROK_URL
from gemini_tool.call_gemini import call_gemini

# === Prompt đầu vào cho LLM Server nội bộ ===
def build_prompt(prompt: str) -> str:
    return (
        "<｜begin▁of▁sentence｜>"
        "<｜system｜>\n"
        "### Hướng dẫn: Hãy là một trợ lý ảo nha khoa và TRÌNH BÀY để trả lời câu hỏi dưới đây:\n"
        "<｜user｜>\n"
        f"### Câu hỏi:\n{prompt.strip()}\n"
    )

# === Format đầu ra chuẩn yêu cầu ===
def format_final_output(answer: str) -> str:
    return (
        "<｜assistant｜>\n"
        "<answer>\n"
        f"# 💬 Câu trả lời\n{answer.strip()}\n"
        "</answer>"
        "<｜end▁of▁sentence｜>"
    )

# === Gửi yêu cầu đến LLM server custom ===
def send_request(prompt: str, generation_params: dict):
    try:
        response = requests.post(NGROK_URL, json={"prompt": prompt, **generation_params}, stream=True)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        return f"Error during generation: {str(e)}"

# === Tạo các prompt phụ cho Gemini để phản biện và cải tiến ===
def build_gemini_prompts(question_prompt: str, raw_answer: str) -> dict:
    return {
        "study_method": f"""Bạn là chuyên gia nha khoa. Với câu hỏi: "{question_prompt}", câu trả lời hiện tại: "{raw_answer}".
Hãy liệt kê chi tiết tất cả các **phương pháp điều trị, kỹ thuật và quy trình** có thể áp dụng. Bao gồm mô tả, khi nào áp dụng, bảng markdown nếu cần.""",

        "fix_presentation": f"""Câu trả lời: "{raw_answer}". Cải thiện cách trình bày cho rõ ràng, chuyên nghiệp hơn. Có thể dùng bảng markdown, danh sách, công thức chuẩn dạng markdown hiển thị hoặc biểu đồ minh họa bằng code python nếu phù hợp.""",

        "knowledge_boost": f"""Câu trả lời: "{raw_answer}". Bổ sung thêm kiến thức giúp học hỏi như:
- Thống kê (có thể gần đúng),
- Tỷ lệ thành công,
- Thời gian điều trị,
- Lưu ý lâm sàng,
- Biến chứng phổ biến.
Có thể tự tạo số liệu gần đúng nhưng hợp lý."""
    }

# === Tổng hợp các phần lại và yêu cầu Gemini trả về câu trả lời hoàn chỉnh đúng định dạng ===
def build_final_synthesis_prompt(question_prompt: str, improved_parts: dict) -> str:
    return f"""
        Bạn là một trợ lý ảo nha khoa. Hãy tổng hợp các phần sau để tạo ra một câu trả lời chuyên sâu, rõ ràng và chuẩn nha khoa.

        Yêu cầu:
        - Trình bày chi tiết, đầy đủ.
        - Dùng bảng markdown nếu phù hợp.
        - Có mã Python vẽ biểu đồ nếu có dữ liệu.
        - Và **phải trả đúng định dạng bên dưới** — không được thay đổi:
        <｜assistant｜>
        <answer>
        💬 Câu trả lời
        [Nội dung trả lời chi tiết ở đây...]
        </answer>
        <｜end▁of▁sentence｜>
        Dưới đây là thông tin bạn cần tổng hợp:
        1. Phương pháp: {improved_parts['study_method']}
        2. Trình bày lại: {improved_parts['fix_presentation']}
        3. Kiến thức bổ sung: {improved_parts['knowledge_boost']}

        Câu hỏi gốc là: {question_prompt}

        ❗**Lưu ý**: Chỉ trả đúng định dạng trên. Không thêm chữ ngoài định dạng.
        """

# === Hàm chính stream kết quả từng dòng ===
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

    # B1: Gửi câu hỏi đến LLM server lấy câu trả lời đầu tiên
    base_prompt = build_prompt(prompt)
    initial_response = send_request(base_prompt, generation_params)
    

    raw_answer = initial_response.text.strip()

    # B2: Tạo 3 prompt phụ và gửi tới Gemini
    sub_prompts = build_gemini_prompts(prompt, raw_answer)
    improved_parts = {}
    for key, sub_prompt in sub_prompts.items():
        try:
            improved_parts[key] = call_gemini(sub_prompt, model_name="models/gemini-2.0-flash")
        except Exception as e:
            improved_parts[key] = f"Lỗi khi gọi Gemini ({key}): {str(e)}"

    # B3: Tạo prompt tổng hợp và gửi tới Gemini để nhận câu trả lời hoàn chỉnh
    try:
        final_synthesis_prompt = build_final_synthesis_prompt(prompt, improved_parts)
        final_answer = call_gemini(final_synthesis_prompt, model_name="models/gemini-2.0-flash")
        formatted_output = final_answer.strip()
    except Exception as e:
        yield f"Error generating final answer: {str(e)}"
        return

    # B4: Trả kết quả ra theo từng dòng
    for line in formatted_output.splitlines(keepends=True):
        yield line