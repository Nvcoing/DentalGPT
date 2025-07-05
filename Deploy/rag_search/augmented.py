from gemini_tool.call_gemini import call_gemini

def augmented(query,context, model_name="models/gemini-1.5-flash-latest"):
    prompt = f"""Dựa trên thông tin truy xuất bên dưới, hãy tóm tắt ngắn gọn và đầy đủ các ý chính có liên quan trực tiếp đến câu truy vấn: "{query}".  
    Nếu thông tin truy xuất không liên quan đến truy vấn, hãy bỏ qua chúng.

    Yêu cầu:
    - Chỉ giữ lại các chi tiết thực sự cần thiết để hiểu và trả lời truy vấn.
    - Ưu tiên thông tin cụ thể, dữ kiện, số liệu nếu có.
    - Lược bỏ nội dung trùng lặp, không liên quan hoặc lan man.
    - Tạo ra một ngữ cảnh súc tích, đầy đủ và hữu ích cho việc trả lời truy vấn.

    Thông tin truy xuất:  
    {context}

    ---

    Sau đó, hãy tạo một prompt trả lời câu truy vấn "{query}" sao cho phù hợp và tối ưu, có sử dụng phần thông tin tóm tắt ở trên (nếu liên quan).  
    Nếu không có thông tin liên quan thì chỉ cần tạo prompt trả lời truy vấn dựa trên kiến thức nền chung.
    """

    summary = call_gemini(prompt, model_name=model_name)
    return summary

# if __name__ == "__main__":
#     query = "Cho tôi bài báo về nha khoa"
#     summary = augmented(query, top_k=3, num_web_results=3)
#     print("📝 Tóm tắt từ Gemini:\n", summary)
