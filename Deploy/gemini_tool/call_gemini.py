import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key từ file .env (chỉ load 1 lần khi import)
load_dotenv()

def get_all_api_keys():
    return [value for key, value in os.environ.items() if key.startswith("GOOGLE_API_KEY_")]

def call_gemini(prompt: str, model_name: str = "models/gemini-1.5-flash-latest") -> str:
    """
    Gọi Gemini API với các key khác nhau cho đến khi thành công.
    """
    api_keys = get_all_api_keys()
    if not api_keys:
        return "Lỗi: Key không tồn tại."

    for key in api_keys:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(model_name=model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            continue
    return "❌ Tất cả các API key đều bị lỗi. Vui lòng kiểm tra lại."

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
