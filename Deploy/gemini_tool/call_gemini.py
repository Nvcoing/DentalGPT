import os
from dotenv import load_dotenv
import google.generativeai as genai

# Luôn load đúng file .env nằm cùng file này
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

def get_all_api_keys():
    return [value for key, value in os.environ.items() if key.startswith("GOOGLE_API_KEY_")]

def call_gemini(prompt: str, model_name: str = "models/gemini-1.5-flash-latest") -> str:
    api_keys = get_all_api_keys()
    
    # Debug: in ra danh sách API key đang có
    print("[DEBUG] Danh sách API key đã được tìm thấy...")
    if not api_keys:
        return "Không thể kết nối LLM 0 (không có API key)."

    # Thử từng key
    for key in api_keys:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(model_name=model_name)
            response = model.generate_content(prompt, stream=True)

            # Ghép kết quả từ streaming
            full_response = ""
            for chunk in response:
                full_response += chunk.text
            return full_response

        except Exception as e:
            print(f"[CẢNH BÁO] Key {key[:10]}... lỗi: {e}")
            continue

    return "Không thể kết nối LLM 0 (tất cả API key đều lỗi)."
