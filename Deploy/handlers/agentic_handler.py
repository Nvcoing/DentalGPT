import requests
import time
from gemini_tool.call_gemini import call_gemini
from config import NGROK_URL

# Template mẫu cho 3 loại báo cáo
TEMPLATES = {
    "report": [
        "Thông tin bệnh nhân",
        "Triệu chứng lâm sàng",
        "Chẩn đoán sơ bộ",
        "Điều trị và lời khuyên"
    ],
    "thesis": [
        "Giới thiệu đề tài",
        "Tổng quan tài liệu",
        "Phương pháp nghiên cứu",
        "Kết quả và thảo luận",
        "Kết luận và kiến nghị"
    ],
    "paper": [
        "Tóm tắt",
        "Giới thiệu",
        "Vật liệu và phương pháp",
        "Kết quả",
        "Thảo luận",
        "Kết luận"
    ]
}

# Xây dựng prompt cơ bản cho LLM
def build_prompt(question: str) -> str:
    return (
        "<｜begin▁of▁sentence｜>"
        "<｜system｜>\n"
        "### Hướng dẫn: Hãy là một trợ lý ảo nha khoa và trình bày để trả lời câu hỏi dưới đây:\n"
        "<｜user｜>\n"
        f"### Câu hỏi:\n{question.strip()}\n"
    )

# Gửi prompt tới server LLM
def send_request(prompt: str, generation_params: dict):
    try:
        response = requests.post(NGROK_URL, json={"prompt": prompt, **generation_params}, stream=True)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        return f"Error during generation: {str(e)}"

# Xác định loại báo cáo từ nội dung đầu vào
def detect_template_type(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if any(kw in prompt_lower for kw in ["báo cáo", "hồ sơ", "phiếu khám"]):
        return "report"
    elif any(kw in prompt_lower for kw in ["luận văn", "luận án", "thesis", "đề tài"]):
        return "thesis"
    elif any(kw in prompt_lower for kw in ["paper", "bài báo khoa học"]):
        return "paper"
    return "report"  # Mặc định

# Hàm chính gọi từ FastAPI
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

    # Xác định loại template và các mục cần sinh
    template_type = detect_template_type(prompt)
    sections = TEMPLATES[template_type]
    results = {}

    # Sinh nội dung từng mục
    for section in sections:
        question = f"{prompt.strip()} - {section}"
        prompt_for_section = build_prompt(question)
        response = send_request(prompt_for_section, generation_params)

        if isinstance(response, str):
            yield response
            return

        output_text = ""
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                decoded = chunk.decode("utf-8")
                output_text += decoded
                yield decoded  # stream ra ngoài cho FastAPI client

        results[section] = output_text.strip()
        time.sleep(0.3)

    # Sau khi xong, tổng hợp lại và gửi cho Gemini để kết luận
    summary_prompt = (
        f"""Bạn là chuyên gia nha khoa. Dưới đây là các mục đã được hoàn thành trong {template_type}:\n\n"""
        + "\n\n".join([f"### {k}:\n{v}" for k, v in results.items()])
        + "\n\nHãy viết một kết luận tổng quan, đánh giá tổng thể và đề xuất cải thiện nếu có."
    )

    final_summary = call_gemini(summary_prompt, model_name="models/gemini-1.5-flash-latest")
    yield "\n\n📌 KẾT LUẬN TỔNG HỢP:\n"
    yield final_summary
    return
