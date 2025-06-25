from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import nltk
# nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize


# === 1. Chia văn bản thành các đoạn nhỏ (mỗi đoạn 2 câu) ===
def chunk_text(text, chunk_size=500, overlap=50):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = " ".join(sentences[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


# === 2. Gọi tool_search và trích xuất văn bản ===
def get_search_chunks(results):
    raw_texts = [item['content'] for item in results if item['content']]
    all_text = "\n".join(raw_texts)
    return chunk_text(all_text)


# === 3. Tạo prompt RAG dựa trên nội dung truy xuất ===
def retrieval(query, results, top_k=5):
    chunks = get_search_chunks(results)
    if not chunks:
        return "Không tìm thấy ngữ cảnh phù hợp.", []

    # Load sentence embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunk_embeddings = model.encode(chunks, convert_to_tensor=False)

    # FAISS Index
    dimension = chunk_embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_embeddings).astype('float32'))

    # Truy vấn
    query_embedding = model.encode([query])[0].astype('float32')
    D, I = index.search(np.array([query_embedding]), top_k)

    retrieved_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(retrieved_chunks)
    return context, retrieved_chunks
