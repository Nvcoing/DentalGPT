from retrieval_search.retrieval import retrieval 
from retrieval_search.google_search_api import tool_search 
from gemini_tool.call_gemini import call_gemini
def augmented(query, model_name="models/gemini-1.5-flash-latest", top_k=5, num_web_results=3):
    results = tool_search(query, num_results=num_web_results)
    context, retrieved_chunks = retrieval(query, results=results, top_k = top_k)
    print(context)
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
#     query = "Trường đại học Phenika ở địa chỉ nào tại Hà Nội"
#     summary = augmented(query, top_k=5, num_web_results=5)
#     print("📝 Tóm tắt từ Gemini:\n", summary)
