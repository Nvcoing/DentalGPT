# train.py
import argparse
from load_model import load_model
from load_dataset import build_dataset
from sft_trainer import get_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--hf_repo", type=str, default="NV9523/DentalGPT")
    parser.add_argument("--hub_token", type=str, required=True)
    args = parser.parse_args()

    model, tokenizer = load_model(model_name=args.model_name)
    train_ds = build_dataset(hf_repo=args.hf_repo)
    trainer = get_trainer(model, tokenizer, train_ds,
                          hub_token=args.hub_token)
    trainer.train()

if __name__ == "__main__":
    main()