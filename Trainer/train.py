import os
from huggingface_hub import snapshot_download
from load_dataset import build_dataset
from load_model import model, tokenizer
from sft_trainer import get_trainer

def find_checkpoint(local_dir: str):
    for root, dirs, _ in os.walk(local_dir):
        for d in dirs:
            if 'checkpoint' in d.lower():
                return os.path.join(root, d)
    return None

if __name__ == '__main__':
    HF_TOKEN = os.getenv('HF_TOKEN')
    REPO = 'NV9523/DentalGPT_SFT'

    # clone once
    local = snapshot_download(repo_id=REPO)
    ckpt = find_checkpoint(local)

    ds = build_dataset()
    trainer = get_trainer(model, tokenizer, ds, REPO, HF_TOKEN)

    if ckpt:
        print('resuming from', ckpt)
        trainer.train(resume_from_checkpoint=ckpt)
    else:
        trainer.train()

    # lưu và push cuối
    model.push_to_hub(REPO, use_temp_dir=False)
    tokenizer.push_to_hub(REPO, use_temp_dir=False)