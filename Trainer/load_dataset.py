# load_dataset.py
from datasets import load_dataset, Dataset
import pandas as pd

def build_dataset(hf_repo: str = "NV9523/DentalGPT") -> Dataset:
    """
    Load dataset from HuggingFace and build prompts.
    """
    ds = load_dataset(hf_repo)['train']

    # rename columns
    ds = ds.rename_columns({
        "Câu hỏi": "question",
        "CoT_Goal": "goal",
        "CoT_Reasoning": "reasoning",
        "CoT_Justification": "justification",
        "Câu trả lời": "answer"
    })

    def create_prompt(row):
        user_prompt = (
            "<|Question|>\n"
            f"Câu hỏi: {row['question']}\n"
            "</|Question|>\n"
            "<|Think|>\n"
            f"Mục tiêu: {row['goal']}\n"
            f"Bước suy nghĩ: {row['reasoning']}\n"
            f"Giải thích: {row['justification']}\n"
            "</|Think|>"
        )
        return f"{user_prompt}\n<|Answer|>\n{row['answer']}</|Answer|>"

    ds = ds.filter(lambda x: not any(v is None for v in x.values()))
    ds = ds.map(lambda row: {'text': create_prompt(row)}, batched=False)
    df = pd.DataFrame(ds)
    return Dataset.from_pandas(df[['text']])