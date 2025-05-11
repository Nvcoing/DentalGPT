from datasets import load_dataset
from datasets import Dataset

def build_dataset(hf_repo: str = "NV9523/DentalGPT") -> Dataset:
    ds = load_dataset(hf_repo, split="train")

    ds = ds.rename_columns({
        "Câu hỏi": "question",
        "CoT_Goal": "goal",
        "CoT_Reasoning": "reasoning",
        "CoT_Justification": "justification",
        "Câu trả lời": "answer"
    })

    def is_valid(example):
        return all(example.get(k) for k in ['question', 'goal', 'reasoning', 'justification', 'answer'])

    ds = ds.filter(is_valid)

    def create_prompt_batch(batch):
        return {
            "text": [
                f"<|Question|>\nCâu hỏi: {q}\n</|Question|>\n<|Think|>\nMục tiêu: {g}\nBước suy nghĩ: {r}\nGiải thích: {j}\n</|Think|>\n<|Answer|>\n{a}</|Answer|>"
                for q, g, r, j, a in zip(batch['question'], batch['goal'], batch['reasoning'], batch['justification'], batch['answer'])
            ]
        }

    ds = ds.map(create_prompt_batch, batched=True, batch_size=64)

    return ds.remove_columns([col for col in ds.column_names if col != "text"])
