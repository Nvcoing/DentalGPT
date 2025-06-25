import os
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document  # ⚠️ Cần để kiểm tra và tạo lại nếu cần

folder_path = r"D:\Folder_HocTap\Đồ án tốt nghiệp\Code\Thesis_FineTune_MoE_ChatBotDental\Build dataset Dental\crawl_Viet_Nam_Medical_Journal\Doccument_Viet_Nam_Medical_Journal"

def load_file(filepath):
    if filepath.endswith(".pdf"):
        loader = PyMuPDFLoader(filepath)
    elif filepath.endswith(".docx"):
        loader = Docx2txtLoader(filepath)
    elif filepath.endswith(".txt"):
        loader = TextLoader(filepath)
    else:
        return []
    return loader.load()

def extract_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.pdf', '.docx', '.txt')):
            filepath = os.path.join(folder_path, filename)
            try:
                loaded_docs = load_file(filepath)
                for doc in loaded_docs:
                    if not hasattr(doc, "page_content"):
                        continue  # Bỏ qua nếu không có nội dung
                    content = doc.page_content.lower()
                    content = ' '.join(content.split())
                    doc.page_content = content
                documents.extend(loaded_docs)
            except Exception as e:
                print(f"Lỗi khi xử lý {filename}: {e}")
    return documents

def chunk_documents(documents, chunk_size=2000, chunk_overlap=200):
    # Đảm bảo toàn bộ `documents` là đối tượng `Document`
    safe_docs = []
    for doc in documents:
        if isinstance(doc, Document):
            safe_docs.append(doc)
        elif isinstance(doc, dict) and "content" in doc:
            safe_docs.append(Document(page_content=doc["content"], metadata=doc.get("metadata", {})))
        else:
            print("Bỏ qua một document không hợp lệ")
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(safe_docs)

def save_to_vector_db(documents, persist_dir="ChromaDB"):
    embedding = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "auto"}
    )
    db = Chroma.from_documents(documents, embedding, persist_directory=persist_dir)
    db.persist()
    print(f"Vector database đã lưu vào thư mục '{persist_dir}'")

# def main():
#     print("Bắt đầu xử lý tài liệu")
#     documents = extract_documents(folder_path)
#     if not documents:
#         print("Không có tài liệu nào được tải")
#         return
#     print(f"Đã trích xuất {len(documents)} tài liệu")

#     chunks = chunk_documents(documents)
#     print(f"Đã chia thành {len(chunks)} đoạn")

#     save_to_vector_db(chunks)

# if __name__ == "__main__":
#     main()
