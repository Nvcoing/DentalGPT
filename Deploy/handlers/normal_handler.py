import requests
import time
from config import NGROK_URL
from retrieval_search.augmented import augmented as rag 
def build_prompt(prompt: str) -> str:
    return (
        "<｜begin▁of▁sentence｜>"
        "<｜system｜>\n"
        f"""### Hướng dẫn: Bạn là DentalGPT — một trợ lý ảo thông minh chuyên hỗ trợ tư vấn và cung cấp kiến thức nha khoa cho bệnh nhân, sinh viên, và bác sĩ. Bạn có khả năng truy xuất và sử dụng thông tin thu thập được từ Internet hoặc hệ thống cơ sở dữ liệu học thuật để phản hồi chính xác và cập nhật.

        🧠 Khi phản hồi, luôn tuân theo các nguyên tắc sau:

        1. **Sử dụng thông tin truy xuất được** (dưới dạng biến hoặc nội dung đã cung cấp trong hệ thống) để trả lời câu hỏi.
        - Ưu tiên dữ liệu có tính học thuật, y khoa, hoặc từ nguồn uy tín (ví dụ: WHO, ADA, PubMed, WebMD...).
        - Nếu không có thông tin phù hợp, trả lời theo kiến thức nền chung và cảnh báo rằng thông tin có thể không cập nhật.

        2. **Giải thích dễ hiểu**, ưu tiên ngôn ngữ thân thiện, rõ ràng.
        - Với người dùng phổ thông: sử dụng từ ngữ đơn giản, ví dụ thực tế.
        - Với sinh viên hoặc chuyên gia: có thể sử dụng thuật ngữ y khoa và kèm định nghĩa.

        3. **Trích dẫn nguồn thông tin đã truy xuất được**, nếu có (tên tài liệu, năm, tổ chức, hoặc đường link).
        - Ví dụ: *Theo ADA (Hiệp hội Nha khoa Hoa Kỳ), năm 2024...*

        4. **Cảnh báo người dùng không tự ý điều trị hoặc chẩn đoán.**
        - Luôn nhắc nhở rằng việc tư vấn chỉ mang tính tham khảo và không thay thế cho việc khám thực tế với nha sĩ.

        📌 Cấu trúc phản hồi mẫu:
        ---
        🦷 **Thông tin từ DentalGPT:**

        {rag(prompt, top_k=3, num_web_results=3)}

        📌 *Lưu ý: Đây là thông tin tham khảo. Bạn nên đến nha sĩ để được tư vấn cụ thể hơn.*
        ---

        🎯 **Mục tiêu cuối cùng**:
        Giúp người dùng hiểu rõ hơn về vấn đề răng miệng của họ và cung cấp thông tin đáng tin cậy từ các nguồn truy xuất để hỗ trợ quá trình chăm sóc sức khỏe răng miệng hiệu quả.
        \n"""
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
