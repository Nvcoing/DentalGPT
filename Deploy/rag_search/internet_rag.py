from keybert import KeyBERT
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from vectordb.build_vectordb import chunk_documents
def extract_keywords_keybert(text, top_n=3, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    kw_model = KeyBERT(model=model_name)
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return [kw for kw, _ in keywords]

def Live_Retrieval_Augmented(
    question: str,
    documents: list,
    top_n_keywords: int = 3,
    top_k_docs: int = 3,
    temperature: float = 0.3,
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    keyword_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    llm_model_name: str = "models/gemini-1.5-flash-latest",
    prompt_template_str: str = None,
    api_key: str = ""
):
    # 1. Từ khóa
    keywords = extract_keywords_keybert(question, top_n=top_n_keywords, model_name=keyword_model_name)
    if not keywords:
        return {"error": "Không tìm thấy từ khóa."}
    keyword_query = " ".join(keywords)

    # 2. Embedding & FAISS
    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cuda"}
    )
    chunked_docs = chunk_documents(documents)
    chunked_texts = [doc.page_content for doc in chunked_docs]
    db = FAISS.from_texts(chunked_texts, embedding)
    retriever = db.as_retriever(search_kwargs={"k": top_k_docs})

    # 3. Prompt template
    default_prompt = f"""
    Bạn là một bác sĩ nha khoa chuyên môn sâu.
    Dựa vào các đoạn tài liệu sau, hãy tạo ngữ cảnh cho chatbot nha khoa dựa vào câu hỏi và từ khóa.

    TÀI LIỆU:
    {retriever}

    CÂU HỎI:
    {keyword_query}
    """
    prompt_str = prompt_template_str if prompt_template_str else default_prompt
    prompt = PromptTemplate.from_template(prompt_str)

    # 4. LLM
    llm = ChatGoogleGenerativeAI(
        model=llm_model_name,
        temperature=temperature,
        google_api_key=api_key
    )

    # 5. QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    # 6. Kết quả
    result = qa_chain({"query": keyword_query})

    return {
        "question": question,
        "keywords": keywords,
        "answer": result["result"],
        "source_documents": [doc.page_content for doc in result["source_documents"]]
    }
# if __name__ == "__main__":
#     documents = [
#         "Viêm nha chu ở trẻ em là tình trạng viêm nhiễm ở mô quanh răng, có thể gây chảy máu, sưng nướu và hôi miệng.",
#         "Các triệu chứng bao gồm: chảy máu chân răng khi đánh răng, nướu sưng đỏ, đau khi ăn uống, và có mảng bám quanh răng.",
#         "Viêm nha chu nếu không điều trị có thể gây mất răng sớm và ảnh hưởng đến sự phát triển xương hàm của trẻ."
#     ]

#     result = Live_Retrieval_Augmented(
#         question="Triệu chứng lâm sàng của bệnh viêm nha chu ở trẻ em?",
#         documents=documents,
#         top_n_keywords=3,
#         top_k_docs=3,
#         temperature=0.3,
#         api_key="API_KEY"
#     )

#     print("🔑 Keywords:", result["keywords"])
#     print("📌 Question:", result["question"])
#     print("🤖 Answer:", result["answer"])
#     print("📚 Sources:")
#     for i, doc in enumerate(result["source_documents"], 1):
#         print(f"[{i}] {doc}")
