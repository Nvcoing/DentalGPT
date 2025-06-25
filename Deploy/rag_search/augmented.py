from gemini_tool.call_gemini import call_gemini

def augmented(query,context, model_name="models/gemini-1.5-flash-latest"):
    prompt = f"""Tóm tắt thông tin sau đây liên quan đến truy vấn: "{query}". 
        Tóm tắt đầy đủ các ý chính liên quan và thông tin thêm cần biết.
        Lưu ý: Tóm tắt thông tin chính bỏ qua các thông tin thừa để ngữ cảnh không quá dài
        --- THÔNG TIN TRUY XUẤT ĐƯỢC ---
        {context}
        --- TẠO NGỮ CẢNH ---"""

    summary = call_gemini(prompt, model_name=model_name)
    return summary

# if __name__ == "__main__":
#     query = "Cho tôi bài báo về nha khoa"
#     summary = augmented(query, top_k=3, num_web_results=3)
#     print("📝 Tóm tắt từ Gemini:\n", summary)
