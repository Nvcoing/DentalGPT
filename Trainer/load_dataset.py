from datasets import load_dataset, Dataset
import pandas as pd

def build_dataset(hf_repo: str = "NV9523/DentalGPT_SFT"):
    # Load dataset từ HuggingFace Hub
    train_ds = load_dataset(hf_repo, split="train")
    eval_ds = load_dataset(hf_repo, split="validation")
    
    # Đổi tên cột theo chuẩn
    def rename_columns(ds):
        ds = ds.rename_columns({
            "Instruction":"instruction",
            "Câu hỏi": "question",
            "CoT_Goal": "goal",
            "CoT_Reasoning": "reasoning",
            "CoT_Justification": "justification",
            "Câu trả lời": "answer",
            "label1":"format",
            "label2":"content",
            "label3":"specialized"
        })
        return ds
    
    # Lọc bỏ các hàng thiếu thông tin
    def is_valid(x):
        return all(x.get(k) for k in ['instruction','question', 'goal', 'reasoning', 'justification', 'answer', "format", "content", "specialized"])
    
    # Process both datasets
    def process_dataset(ds):
        ds = rename_columns(ds)
        ds = ds.filter(is_valid)
        return ds
    
    train_ds = process_dataset(train_ds)
    eval_ds = process_dataset(eval_ds)
    
    # Hàm tạo prompt theo định dạng mới
    def create_prompt(batch):
        prompts = []
        for i, q, g, r, j, a, f, c, s in zip(
            batch['instruction'], batch['question'], batch['goal'], 
            batch['reasoning'], batch['justification'], batch['answer'],
            batch['format'], batch['content'], batch['specialized']
        ):
            prompt = (
                "<｜begin▁of▁sentence｜>"
                "<｜system｜>\n"
                f"### Hướng dẫn: {i.strip()}\n"
                "<｜user｜>\n"
                f"### Câu hỏi:\n {q.strip()}\n"
                "<｜think｜>\n"
                "Hãy cùng diễn giải từng bước nào!🤔\n"
                "<reasoning_cot>\n"
                "# 🧠 Suy luận của DentalGPT\n"
                f"## 1️⃣ Mục tiêu 📌\n{g.strip()}\n"
                f"## 2️⃣ Bước suy nghĩ ⚙️\n{r.strip()}\n"
                f"## 3️⃣ Giải thích 📝\n{j.strip()}\n"
                "</reasoning_cot>\n"
                "<｜expert｜>\n"
                "<experting>\n"
                "# 👨‍🔬 Chuyên gia\n"
                f"## Trình bày dạng: {f.strip()}\n"
                f"## Nội dung về: {c.strip()}\n"
                f"## Chuyên sâu về: {s.strip()}\n"
                "</experting>\n"
                "<｜assistant｜>\n"
                "<answer>\n"
                f"# 💬 Câu trả lời\n{a.strip()}\n"
                "</answer>"
                "<｜end▁of▁sentence｜>"
            )
            prompts.append(prompt)
        return {"text": prompts}
    
    # Apply to both datasets
    train_ds = train_ds.map(create_prompt, batched=True, batch_size=1024)
    eval_ds = eval_ds.map(create_prompt, batched=True, batch_size=1024)
    
    return train_ds.remove_columns([col for col in train_ds.column_names if col != "text"]), \
           eval_ds.remove_columns([col for col in eval_ds.column_names if col != "text"])