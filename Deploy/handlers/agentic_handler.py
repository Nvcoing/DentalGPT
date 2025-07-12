import requests
import time
import json
from gemini_tool.call_gemini import call_gemini
from config import NGROK_URL
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(CURRENT_DIR, "templates.json")
# Tải template từ JSON
with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    TEMPLATES = json.load(f)

def build_prompt(question: str) -> str:
    """Xây dựng prompt cơ bản cho LLM"""
    return (
        "<｜begin▁of▁sentence｜>"
        "<｜system｜>\n"
        "### Hướng dẫn: Hãy là là một trợ lý ảo nha khoa và trả lời câu hỏi dưới đây:\n"
        "<｜user｜>\n"
        f"### Câu hỏi:\n{question.strip()}\n"
        # "<｜think｜>\n"
        # "Hãy cùng diễn giải từng bước nào!🤔\n"
        # "<reasoning_cot>\n"
        # "# 🧠 Suy luận của DentalGPT\n"
        # f"## 1️⃣ Mục tiêu 📌\nTrả lời đơn giản, đúng trọng tâm, ngắn gọn, dễ hiểu\n"
        # f"## 2️⃣ Bước suy nghĩ ⚙️\nBước 1: Xác định đúng câu hỏi\nBước 2: Xác định câu trả lời\nBước 3: Xác định cách trình bày\n"
        # f"## 3️⃣ Giải thích 📝\nGiải thích ngắn gọn\n"
        # "</reasoning_cot>\n"
    )

def send_request(prompt: str, generation_params: dict):
    """Gửi prompt tới server LLM"""
    try:
        response = requests.post(NGROK_URL, json={"prompt": prompt, **generation_params}, stream=True)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        return f"Error during generation: {str(e)}"

def parse_llm_response(raw_response: str) -> str:
    """Parse response từ LLM để lấy phần <answer>"""
    try:
        # Tìm phần <answer>
        start_tag = "<answer>"
        end_tag = "</answer>"
        
        start_idx = raw_response.find(start_tag)
        end_idx = raw_response.find(end_tag)
        
        if start_idx != -1 and end_idx != -1:
            # Lấy nội dung giữa <answer> và </answer>
            content = raw_response[start_idx + len(start_tag):end_idx].strip()
            return content
        else:
            # Nếu không tìm thấy tag, trả về toàn bộ response
            return raw_response.strip()
    except Exception as e:
        return raw_response.strip()

def detect_template_type(prompt: str) -> str:
    """Xác định loại báo cáo từ nội dung đầu vào"""
    prompt_lower = prompt.lower()
    if any(kw in prompt_lower for kw in ["báo cáo", "hồ sơ", "phiếu khám", "bệnh án", "khám bệnh"]):
        return "report"
    elif any(kw in prompt_lower for kw in ["luận văn", "luận án", "thesis", "đề tài", "đồ án"]):
        return "thesis"
    elif any(kw in prompt_lower for kw in ["paper", "bài báo", "nghiên cứu khoa học", "journal"]):
        return "paper"
    return "report"  # Mặc định

def generate_section_questions(user_prompt: str, section_title: str, section_description: str, template_type: str) -> str:
    """Sử dụng Gemini để tạo câu hỏi chi tiết cho từng mục"""
    
    gemini_prompt = f"""
        Bạn là chuyên gia nha khoa có kinh nghiệm. Tôi đang viết một {TEMPLATES[template_type]['name']}.

        Yêu cầu của người dùng: "{user_prompt}"

        Mục hiện tại: "{section_title}"
        Mô tả mục: "{section_description}"

        Hãy tạo ra 3-5 câu hỏi chi tiết và cụ thể để hướng dẫn viết nội dung cho mục này. 
        Các câu hỏi cần:
        1. Phù hợp với yêu cầu của người dùng
        2. Tập trung vào mục "{section_title}" 
        3. Có tính chuyên môn cao trong lĩnh vực nha khoa
        4. Giúp tạo ra nội dung chất lượng và đầy đủ

        Chỉ trả về danh sách câu hỏi, mỗi câu hỏi trên một dòng bắt đầu bằng "- ".
        """
            
    try:
        questions = call_gemini(gemini_prompt, model_name="models/gemini-1.5-flash-latest")
        return questions.strip()
    except Exception as e:
        # Fallback questions nếu Gemini lỗi
        return f"- Hãy mô tả chi tiết về {section_title} trong bối cảnh: {user_prompt}"

def format_output_section(section_title: str, questions: str, content: str) -> str:
    """Format output cho từng mục"""
    return f"""
        ## {section_title}

        ### Câu hỏi hướng dẫn:
        {questions}

        ### Nội dung:
        {content}

        ---
        """

def create_final_report_with_gemini(template_name: str, user_prompt: str, results: dict) -> str:
    """Sử dụng Gemini để tổng hợp template thành báo cáo cuối cùng"""
    
    # Chuẩn bị nội dung để gửi cho Gemini
    sections_content = ""
    for section_title, data in results.items():
        sections_content += f"**{section_title}:**\n{data['content']}\n\n"
    
    gemini_prompt = f"""
        Bạn là chuyên gia nha khoa có kinh nghiệm. Tôi có một {template_name} với các mục đã được hoàn thành như sau:

        **Yêu cầu ban đầu:** {user_prompt}

        **Nội dung các mục:**
        {sections_content}

        Nhiệm vụ của bạn:
        1. Tổng hợp toàn bộ nội dung thành một báo cáo hoàn chỉnh, mạch lạc
        2. Đảm bảo tính liên kết giữa các mục
        3. Bổ sung thêm thông tin cần thiết nếu có
        4. Sửa lỗi chính tả, ngữ pháp nếu có
        5. Định dạng lại cho chuyên nghiệp và dễ đọc
        6. Thêm các khuyến nghị cụ thể và thực tế
        7. **Nếu có dữ liệu định lượng, hãy biểu diễn chúng bằng biểu đồ (với mã Python matplotlib)**
        8. **Nếu có thông tin định tính cần hệ thống, hãy trình bày bằng bảng (dùng markdown table)**
        9. **Nếu có các khái niệm y học liên quan, hãy trình bày thêm công thức hoặc biểu thức dưới dạng markdown chuẩn**

        Hãy viết lại toàn bộ báo cáo theo cấu trúc chuẩn, đảm bảo:
        - Ngôn ngữ chuyên nghiệp nhưng dễ hiểu
        - Thông tin chính xác và khoa học
        - Cấu trúc rõ ràng với các tiêu đề phù hợp
        - Nội dung đầy đủ và toàn diện
        - Bao gồm biểu đồ minh họa (nếu có), bảng thông tin, và công thức (nếu phù hợp)

        Chỉ trả về nội dung báo cáo cuối cùng, không cần giải thích thêm.
    """

    
    try:
        final_report = call_gemini(gemini_prompt, model_name="models/gemini-2.5-flash")
        return final_report.strip()
    except Exception as e:
        # Fallback nếu Gemini lỗi
        fallback_report = f"# {template_name}\n\n"
        fallback_report += f"**Yêu cầu:** {user_prompt}\n\n"
        for section_title, data in results.items():
            fallback_report += f"## {section_title}\n\n{data['content']}\n\n"
        fallback_report += f"\n⚠️ Lưu ý: Báo cáo này chưa được tổng hợp bởi AI do lỗi kỹ thuật: {str(e)}"
        return fallback_report

def generate_response(prompt: str,
                      temperature=0.1, top_p=0.9, top_k=50,
                      repetition_penalty=1.0, do_sample=True,
                      max_new_tokens=1024):
    """Hàm chính gọi từ FastAPI - Tạo báo cáo theo template"""

    generation_params = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample
    }

    try:
        # Xác định loại template
        template_type = detect_template_type(prompt)
        template = TEMPLATES[template_type]
        
        # Tạo full response thay vì stream
        full_response = ""
        
        # Header
        full_response += f"# {template['name']}\n\n"
        full_response += f"**Yêu cầu:** {prompt}\n\n"
        full_response += "=" * 50 + "\n\n"

        results = {}
        
        # Sinh nội dung từng mục
        for i, section in enumerate(template['sections'], 1):
            section_title = section['title']
            section_description = section['description']
            
            full_response += f"📝 **Đang xử lý mục {i}/{len(template['sections'])}: {section_title}**\n\n"
            
            # Bước 1: Tạo câu hỏi với Gemini
            full_response += "🤖 Tạo câu hỏi hướng dẫn...\n"
            questions = generate_section_questions(prompt, section_title, section_description, template_type)
            
            # Bước 2: Tạo prompt chi tiết cho LLM
            detailed_prompt = f"""
                Yêu cầu gốc: {prompt}

                Mục cần viết: {section_title}
                Mô tả: {section_description}

                Câu hỏi hướng dẫn:
                {questions}

                Hãy viết nội dung chi tiết và chuyên nghiệp cho mục "{section_title}" dựa trên các câu hỏi hướng dẫn trên.
                Nội dung cần có cấu trúc rõ ràng, sử dụng thuật ngữ chuyên môn phù hợp và đảm bảo tính khoa học.
                """
            
            # Bước 3: Gửi tới LLM
            full_response += "⚡ Sinh nội dung từ LLM...\n"
            llm_prompt = build_prompt(detailed_prompt)
            response = send_request(llm_prompt, generation_params)
            
            if isinstance(response, str):
                full_response += f"❌ Lỗi: {response}\n"
                continue
                
            # Thu thập response từ LLM
            section_content = ""
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    decoded = chunk.decode("utf-8")
                    section_content += decoded
            
            # Parse để lấy phần <answer>
            parsed_content = parse_llm_response(section_content)
            
            # Lưu kết quả
            results[section_title] = {
                'questions': questions,
                'content': parsed_content
            }
            
            # Add thông tin tạm thời (sẽ được thay thế bằng báo cáo cuối)
            full_response += f"✅ **Hoàn thành mục: {section_title}**\n"
            full_response += f"Nội dung: {len(parsed_content)} ký tự\n\n"
            time.sleep(0.3)  # Tránh spam

        # Tạo báo cáo cuối cùng với Gemini
        full_response += "🔄 **Đang tổng hợp báo cáo cuối cùng bằng Gemini...**\n\n"
        
        try:
            final_report = create_final_report_with_gemini(template['name'], prompt, results)
            
            # Thay thế toàn bộ nội dung bằng báo cáo cuối
            full_response = "=" * 60 + "\n"
            full_response += "🎯 **BÁO CÁO CUỐI CÙNG - ĐÃ ĐƯỢC TỔNG HỢP BỞI AI**\n"
            full_response += "=" * 60 + "\n\n"
            full_response += final_report
            full_response += "\n\n" + "=" * 60 + "\n"
            full_response += "✅ **Hoàn thành báo cáo!**\n"
            full_response += f"📊 **Thống kê:** Đã xử lý {len(template['sections'])} mục"
            
        except Exception as e:
            full_response += f"\n❌ Lỗi tổng hợp báo cáo cuối: {str(e)}\n"
            full_response += "\n📋 **BÁO CÁO GỐC (chưa tổng hợp):**\n\n"
            for section_title, data in results.items():
                full_response += f"## {section_title}\n\n{data['content']}\n\n---\n\n"
        
        # Return full response as string để main.py có thể xử lý
        return full_response
        
    except Exception as e:
        return f"❌ Lỗi tổng thể: {str(e)}"