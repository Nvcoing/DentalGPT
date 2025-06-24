import os
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

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
                    content = doc.page_content.lower()
                    content = ' '.join(content.split())
                    doc.page_content = content
                documents.extend(loaded_docs)
            except Exception as e:
                print(f"Loi khi xu ly {filename}: {e}")
    return documents

def chunk_documents(documents, chunk_size=5000, chunk_overlap=500):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def save_to_vector_db(documents, persist_dir="ChromaDB"):
    embedding = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "auto"}
    )
    db = Chroma.from_documents(documents, embedding, persist_directory=persist_dir)
    db.persist()
    print(f"Vector database da luu vao thu muc '{persist_dir}'")

def main():
    print("Bat dau xu ly tai lieu")
    documents = extract_documents(folder_path)
    if not documents:
        print("Khong co tai lieu nao duoc tai")
        return
    print(f"Da trich xuat {len(documents)} tai lieu")

    chunks = chunk_documents(documents)
    print(f"Da chia thanh {len(chunks)} doan")

    save_to_vector_db(chunks)

if __name__ == "__main__":
    main()
