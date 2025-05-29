from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd

def build_dataset(hf_repo: str = "NV9523/DentalGPT_SFT"):
    # Load dataset từ HuggingFace Hub
    ds = load_dataset(hf_repo, split="train")
    
    # Chuyển sang pandas DataFrame để xử lý
    df = ds.to_pandas()
    
    # Tạo eval dataset bằng cách lấy mẫu theo nhóm
    eval_df = df.groupby(['label1', 'label2', 'label3']).head(10).reset_index(drop=True)
    
    # Loại bỏ các mẫu eval từ train dataset
    train_df = df[~df.index.isin(eval_df.index)]
    
    # Đổi tên cột theo chuẩn
    def rename_columns(df):
        return df.rename(columns={
            "Instruction": "instruction",
            "Câu hỏi": "question",
            "CoT_Goal": "goal",
            "CoT_Reasoning": "reasoning",
            "CoT_Justification": "justification",
            "Câu trả lời": "answer",
            "label1": "format",
            "label2": "content",
            "label3": "specialized"
        })
    
    train_df = rename_columns(train_df)
    eval_df = rename_columns(eval_df)
    
    # Lọc bỏ các hàng thiếu thông tin
    def is_valid(row):
        return all(row.get(k) for k in ['instruction', 'question', 'goal', 
                                      'reasoning', 'justification', 'answer',
                                      'format', 'content', 'specialized'])
    
    train_df = train_df[train_df.apply(is_valid, axis=1)]
    eval_df = eval_df[eval_df.apply(is_valid, axis=1)]
    
    # Tạo prompt
    def create_prompt(row):
        return (
            "<|system|>\n"
            f"###Hướng dẫn: {row['instruction'].strip()}\n"
            "<|user|>\n"
            f"###Câu hỏi:\n {row['question'].strip()}\n"
            "<|think|>\n"
            "Hãy cùng diễn giải từng bước nào!🤔\n"
            "<reasoning_cot>\n"
            "# 🧠 Suy luận của DentalGPT\n"
            f"## 1️⃣ Mục tiêu 📌\n{row['goal'].strip()}\n"
            f"## 2️⃣ Bước suy nghĩ ⚙️\n{row['reasoning'].strip()}\n"
            f"## 3️⃣ Giải thích 📝\n{row['justification'].strip()}\n"
            "</reasoning_cot>\n"
            "<|expert|>\n"
            "<experting>\n"
            "# 👨‍🔬 Chuyên gia\n"
            f"##Trình bày dạng: {row['format'].strip()}\n"
            f"##Nội dung về: {row['content'].strip()}\n"
            f"##Chuyên sâu về: {row['specialized'].strip()}\n"
            "</experting>\n"
            "<|assistant|>\n"
            "<answer>\n"
            f"# 💬 Câu trả lời\n{row['answer'].strip()}\n"
            "</answer>"
        )
    
    train_df['text'] = train_df.apply(create_prompt, axis=1)
    eval_df['text'] = eval_df.apply(create_prompt, axis=1)
    
    # Chuyển lại sang Dataset
    train_ds = Dataset.from_pandas(train_df[['text']])
    eval_ds = Dataset.from_pandas(eval_df[['text']])
    
    return train_ds, eval_ds