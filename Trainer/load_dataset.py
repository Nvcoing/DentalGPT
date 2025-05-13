from datasets import load_dataset, Dataset

def build_dataset(hf_repo: str = "NV9523/DentalGPT_SFT", filename: str = "Dental_CoT_dataset.parquet") -> Dataset:
    ds = load_dataset(hf_repo, data_files=filename, split="train")
    ds = ds.rename_columns({
        "Câu hỏi": "question",
        "CoT_Goal": "goal",
        "CoT_Reasoning": "reasoning",
        "CoT_Justification": "justification",
        "Câu trả lời": "answer"
    })

    def valid(x):
        return all(x.get(k) for k in ['question','goal','reasoning','justification','answer'])
    ds = ds.filter(valid)

    def make_prompt(batch):
        texts = []
        for q,g,r,j,a in zip(batch['question'], batch['goal'], batch['reasoning'], batch['justification'], batch['answer']):
            prompt = (
                "<|Question|>\n"
                f"Câu hỏi: {q}\n"
                "</|Question|>\n"
                "<|Think|>\n"
                f"Mục tiêu: {g}\n"
                f"Bước suy nghĩ: {r}\n"
                f"Giải thích: {j}\n"
                "</|Think|>\n"
                "<|Answer|>\n"
                f"{a}</|Answer|>"
            )
            texts.append(prompt)
        return {"text": texts}

    ds = ds.map(make_prompt, batched=True, batch_size=64)
    return ds.remove_columns([c for c in ds.column_names if c != 'text'])