from keybert import KeyBERT
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def extract_keywords_keybert(text, top_n=5):
    kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return [kw for kw, _ in keywords]

def run_keybert_qa(query, persist_dir="ChromaDB", top_k=5):
    # 1. Trích xuất từ khóa
    keywords = extract_keywords_keybert(query)
    print("Từ khóa:", ", ".join(keywords))
    if not keywords:
        print("Không tìm thấy từ khóa.")
        return

    keyword_query = " ".join(keywords)

    # 2. Tải Embedding và database
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )
    db = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    retriever = db.as_retriever(search_kwargs={"k": top_k})

    # 3. Truy xuất tài liệu dựa vào keyword_query
    retrieved_docs = retriever.get_relevant_documents(keyword_query)

    return {
        "question": query,
        "keywords": keywords,
        "document": [doc.page_content for doc in retrieved_docs]
    }


# if __name__ == "__main__":
#     run_keybert_qa("Cho tôi biết một số bài báo về nhồi máu cơ tim")
