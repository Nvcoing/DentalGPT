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

    # Hàm tạo prompt theo định dạng mới
    def create_prompt(batch):
        prompts = []
        for q, g, r, j, a in zip(batch['question'], batch['goal'], batch['reasoning'], batch['justification'], batch['answer']):
            prompt = (
                "<｜begin▁of▁sentence｜>"
                "<｜user｜>\n"
                f"###Câu hỏi:\n {q.strip()}\n"
                "<|think|>\n"
                "Hãy cùng diễn giải từng bước nào!🤔\n"
                "<reasoning_cot>\n"
                "# 🧠 Suy luận của DentalGPT\n"
                f"## 1️⃣ Mục tiêu 📌\n{g.strip()}\n"
                f"## 2️⃣ Bước suy nghĩ ⚙️\n{r.strip()}\n"
                f"## 3️⃣ Giải thích 📝\n{j.strip()}\n"
                "</reasoning_cot>\n"
                "<|assistant|>\n"
                "<answer>\n"
                f"# 💬 Câu trả lời\n{a.strip()}\n"
                "</answer>"
                "<｜end▁of▁sentence｜>"
            )
            prompts.append(prompt)
        return {"text": prompts}

    # Tạo cột 'text'
    ds = ds.map(create_prompt, batched=True, batch_size=64)

    # Chỉ giữ lại cột 'text' trong dataset
    return ds.remove_columns([col for col in ds.column_names if col != "text"])
