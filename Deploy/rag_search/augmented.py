from gemini_tool.call_gemini import call_gemini

def augmented(query,context, model_name="models/gemini-1.5-flash-latest"):
    prompt = f"""Tóm tắt ngắn gọn, đầy đủ các ý chính có liên quan đến câu truy vấn: "{query}".
    Hãy tạo prompt ngắn gọn nhưng đầy đủ các ý chính để trả lời cho câu truy vấn dựa trên thông tin truy xuất.

    Yêu cầu:
    - Chỉ giữ lại những thông tin thật sự cần thiết để hiểu và trả lời truy vấn.
    - Loại bỏ nội dung trùng lặp, lan man hoặc không liên quan.
    - Ưu tiên các chi tiết cụ thể, dữ kiện, số liệu (nếu có).
    - Mục tiêu là tạo ra một ngữ cảnh súc tích nhưng vẫn đủ thông tin cần thiết.

    Thông tin truy xuất:
    {context}

    Prompt:"""

    summary = call_gemini(prompt, model_name=model_name)
    return summary

# if __name__ == "__main__":
#     query = "Cho tôi bài báo về nha khoa"
#     summary = augmented(query, top_k=3, num_web_results=3)
#     print("📝 Tóm tắt từ Gemini:\n", summary)
