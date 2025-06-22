from .retrieval import retrieval 
from .google_search_api import tool_search 
from gemini_tool.call_gemini import call_gemini
def format_type(query):
    non_html_types = [
        'pdf', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx', 'csv',
        'txt', 'rtf', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg',
        'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv', 'webm'
    ]

    formats_str = ', '.join(non_html_types)
    prompt = f"""Bạn là một trợ lý thông minh có nhiệm vụ xác định định dạng dữ liệu người dùng cần truy xuất từ internet, dựa trên truy vấn của họ. Chỉ chọn **một định dạng duy nhất** phù hợp nhất với yêu cầu trong số các định dạng sau:

    {formats_str}

    Hãy trả lời bằng **chỉ một từ** là định dạng (ví dụ: pdf, png, mp4). Nếu không thấy phù hợp, trả lời: `html`.

    Dưới đây là truy vấn người dùng:
    "{query}"
    """
    return prompt
def augmented(query, model_name="models/gemini-1.5-flash-latest", top_k=5, num_web_results=3):
    file_type = format_type(query)
    file_type = call_gemini(file_type, model_name=model_name)
    results = tool_search(query, num_results=num_web_results)
    context, retrieved_chunks = retrieval(query, results=results, top_k = top_k)
    if not retrieved_chunks:
        return "Không tìm thấy thông tin phù hợp để tóm tắt."

    prompt = f"""Tóm tắt thông tin sau đây liên quan đến truy vấn: "{query}". 
        Tóm tắt đầy đủ các ý chính liên quan và thông tin thêm cần biết.

        --- NGỮ CẢNH ---
        {context}
        --- HẾT NGỮ CẢNH ---

        Trả lại phần tóm tắt ngắn gọn và dễ hiểu:"""

    summary = call_gemini(prompt, model_name=model_name)
    return summary

# if __name__ == "__main__":
#     query = "Cho tôi bài báo về nha khoa"
#     summary = augmented(query, top_k=3, num_web_results=3)
#     print("📝 Tóm tắt từ Gemini:\n", summary)
