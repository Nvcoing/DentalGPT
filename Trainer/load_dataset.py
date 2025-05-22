from datasets import load_dataset, Dataset

def build_dataset(hf_repo: str = "NV9523/DentalGPT_SFT", filename: str = "Dental_CoT_dataset.parquet") -> Dataset:
    # Load dataset từ HuggingFace Hub
    ds = load_dataset(hf_repo, data_files=filename, split="train")

    # Đổi tên cột theo chuẩn
    ds = ds.rename_columns({
        "Câu hỏi": "question",
        "CoT_Goal": "goal",
        "CoT_Reasoning": "reasoning",
        "CoT_Justification": "justification",
        "Câu trả lời": "answer"
    })

    # Lọc bỏ các hàng thiếu thông tin
    def is_valid(x):
        return all(x.get(k) for k in ['question', 'goal', 'reasoning', 'justification', 'answer'])
    ds = ds.filter(is_valid)

    # Tạo chat records theo format template
    def make_chat_records(batch):
        records = []
        for q, g, r, j, a in zip(batch['question'], batch['goal'], batch['reasoning'], batch['justification'], batch['answer']):
            messages = [
                {"role": "system", "content": "Bạn là trợ lý nha khoa thông minh. Trước khi trả lời, hãy trình bày suy luận đầy đủ."},
                {"role": "user", "content": q.strip()},
                {"role": "assistant", "content": (
                    "Hãy cùng diễn giải từng bước nào!🤔\n"
                    "<reasoning_cot>\n"
                    "# 🧠 Suy luận của DentalGPT\n"
                    f"## 1️⃣ Mục tiêu 📌\n{g.strip()}\n"
                    f"## 2️⃣ Bước suy nghĩ ⚙️\n{r.strip()}\n"
                    f"## 3️⃣ Giải thích 📝\n{j.strip()}\n"
                    "</reasoning_cot>\n"
                    "<answer>\n"
                    f"# 💬 Câu trả lời\n{a.strip()}\n"
                    "</answer>"
                )}
            ]
            records.append({"messages": messages})
        return {"messages": records}

    ds = ds.map(make_chat_records, batched=True, batch_size=64, remove_columns=ds.column_names)

    # ds bây giờ có cột "messages" chứa list of dict như mẫu, đổi dạng Dataset.from_list nếu cần:
    # Nhưng map giữ nguyên định dạng Dataset, có thể flatten cột messages nếu muốn
    # hoặc giữ nguyên để dùng trực tiếp với chat template tokenizer

    return ds
