import os
import argparse
import pandas as pd
from datasets import Dataset

from load_model import load_model_and_tokenizer
from orpo_trainer import create_orpo_trainer
import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

def main(hf_token):
    # Đăng nhập HF
    os.system(f"huggingface-cli login --token {hf_token}")

    # Load model/tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Tải và xử lý dữ liệu RLHF
    df = pd.read_parquet("hf://datasets/NV9523/DentalGPT_SFT/RLHF/RLHF.parquet")
    df = df.dropna(subset=["prompt", "accepted", "rejected"])
    df = df.rename(columns={"accepted": "chosen", "rejected": "rejected", "prompt": "prompt"})
    dataset_pref = Dataset.from_pandas(df)

    assert all(col in dataset_pref.column_names for col in ['prompt', 'chosen', 'rejected'])

    # Tạo trainer
    trainer = create_orpo_trainer(model, tokenizer, dataset_pref)

    # Train
    trainer.train()

    # Push lên HF Hub
    model.push_to_hub("NV9523/DentalGPT_RLHF")
    tokenizer.push_to_hub("NV9523/DentalGPT_RLHF")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="Hugging Face token")
    args = parser.parse_args()
    main(args.token)
