# !python Trainer/train.py \
#   --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#   --hf_repo_dataset NV9523/DentalGPT \
#   --hf_repo_model NV9523/DentalGPT_SFT\
#   --hub_token HF_Token\
#   --resume_from_hf

import argparse
import os
from load_model import load_model
from load_dataset import build_dataset
from sft_trainer import get_trainer
from huggingface_hub import snapshot_download, login


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--hf_repo_dataset", type=str, default="NV9523/DentalGPT")
    parser.add_argument("--hf_repo_model", type=str, default="NV9523/DentalGPT_SFT", help="Huggingface repository model name")
    parser.add_argument("--hub_token", type=str, required=True)
    parser.add_argument("--resume_from_hf", action="store_true", help="Resume from latest checkpoint on HF Hub")
    args = parser.parse_args()

    # Đăng nhập vào Hugging Face
    login(token=args.hub_token)

    # Biến để lưu đường dẫn tải checkpoint từ HF
    repo_dir = None

    # Tải toàn bộ repo từ HF nếu được chỉ định
    if args.resume_from_hf:
        print(f"Downloading entire repo from {args.hf_repo_model}...")
        repo_dir = snapshot_download(
            repo_id=args.hf_repo_model,
            token=args.hub_token,
            allow_patterns=["**"]  # Tải tất cả các file trong repo
        )

        print(f"Repo downloaded to: {repo_dir}")

    # Load model & data
    model, tokenizer = load_model(model_name=args.model_name)
    train_ds = build_dataset(hf_repo=args.hf_repo_dataset)

    # Get trainer
    trainer = get_trainer(model, tokenizer, train_ds,
                          hub_token=args.hub_token,
                          hub_model_id=args.hf_repo_model)

    # Nếu có đường dẫn repo (tải từ HF), thiết lập để resume từ checkpoint
    if repo_dir:
        checkpoint_dir = os.path.join(repo_dir, "checkpoint")
        if os.path.isdir(checkpoint_dir):
            print(f"Found checkpoint folder at {checkpoint_dir}, resuming from there.")
            trainer.train(resume_from_checkpoint=checkpoint_dir)
        else:
            print(f"No checkpoint found at {checkpoint_dir}. Starting training from scratch.")
            trainer.train()
    else:
        trainer.train()


if __name__ == "__main__":
    main()

