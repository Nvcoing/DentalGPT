import os
import argparse
from huggingface_hub import snapshot_download
from load_dataset import build_dataset
from load_model import model, tokenizer
from sft_trainer import get_trainer
import torch

def find_checkpoint(local_dir: str):
    for root, dirs, _ in os.walk(local_dir):
        for d in dirs:
            if 'checkpoint' in d.lower():
                return os.path.join(root, d)
    return None

def parse_args():
    parser = argparse.ArgumentParser(description='Train DentalGPT model')
    parser.add_argument('--hf_token', type=str, required=True, help='HuggingFace access token')
    parser.add_argument('--wandb_key', type=str, required=True, help='Weights & Biases API key')
    parser.add_argument('--repo', type=str, default='NV9523/DentalGPT_SFT', help='HuggingFace repository ID')
    parser.add_argument('--eval_samples', type=int, default=10, help='Number of samples per group for evaluation')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # clone once
    local = snapshot_download(repo_id=args.repo, token=args.hf_token)
    ckpt = find_checkpoint(local)

    train_ds, eval_ds = build_dataset(args.repo)
    trainer = get_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        repo_id=args.repo,
        hf_token=args.hf_token,
        wandb_key=args.wandb_key
    )

    if ckpt:
        print('Resuming from checkpoint:', ckpt)
        # Patch the Trainer's _load_rng_state method to use weights_only=False
        original_load_rng_state = trainer._load_rng_state
        def patched_load_rng_state(checkpoint):
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if os.path.isfile(rng_file):
                return torch.load(rng_file, weights_only=False)
            return None
        trainer._load_rng_state = patched_load_rng_state
        
        trainer.train(resume_from_checkpoint=ckpt)
    else:
        trainer.train()

    # Save and push final model
    model.push_to_hub(args.repo, use_temp_dir=False, token=args.hf_token)
    tokenizer.push_to_hub(args.repo, use_temp_dir=False, token=args.hf_token)