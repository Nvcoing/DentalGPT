import os
import google.generativeai as genai
# from dotenv import load_dotenv

# Load API key từ file .env (chỉ load 1 lần khi import)
# load_dotenv()
# API_KEY = os.getenv("GOOGLE_API_KEY")
API_KEY="AIzaSyDGA68vEoOeUV2ejZ2Epw83i89nK8wgzPo"
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY không được tìm thấy trong file .env")

# Cấu hình Gemini
genai.configure(api_key=API_KEY)

def call_gemini(prompt: str, model_name: str = "models/gemini-1.5-flash-latest") -> str:
    """
    Hàm gọi Gemini API với prompt và model cụ thể.
    Trả về phản hồi dạng chuỗi.
    """
    try:
        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Lỗi: {e}"


# Nếu chạy trực tiếp từ dòng lệnh
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tool gọi Gemini từ prompt.")
    parser.add_argument("--prompt", required=True, help="Nội dung prompt gửi tới Gemini.")
    parser.add_argument("--model", default="models/gemini-1.5-flash-latest", help="Tên model Gemini.")

    args = parser.parse_args()

    result = call_gemini(args.prompt, args.model)
    print("📤 Prompt:", args.prompt)
    print("📥 Gemini trả lời:\n", result)
# from gemini_tool.call_gemini import call_gemini

# response = call_gemini("Tóm tắt mô hình CNN là gì?", model_name="models/gemini-1.5-pro-latest")
# print(response)
