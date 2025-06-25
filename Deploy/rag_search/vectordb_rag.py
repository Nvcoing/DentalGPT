from keybert import KeyBERT
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from gemini_tool.call_gemini import get_all_api_keys as gemini_key
for k in gemini_key():
    key = k

def extract_keywords_keybert(text, top_n=5):
    kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return [kw for kw, _ in keywords]

def run_keybert_qa(query, persist_dir="chroma_db", top_k=5):
    keywords = extract_keywords_keybert(query)
    print("Từ khóa:", ", ".join(keywords))
    if not keywords:
        print("Không tìm thấy từ khóa.")
        return

    keyword_query = " ".join(keywords)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )
    db = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    retriever = db.as_retriever(search_kwargs={"k": top_k})

    prompt_template = """
    Bạn là một bác sĩ chuyên môn về nha khoa và y dược.
    Dựa vào các đoạn tài liệu sau, hãy tạo phản hồi phù hợp cho câu hỏi sau bằng cách sử dụng từ khóa.

    TÀI LIỆU:
    {context}

    CÂU HỎI:
    {question}
    """
    prompt = PromptTemplate.from_template(prompt_template)

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest",
        temperature=0.3,
        google_api_key=key
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    result = qa_chain({"query": keyword_query})

    return result["result"]

# if __name__ == "__main__":
#     run_keybert_qa("Cho tôi biết một số bài báo về nhồi máu cơ tim")
