import subprocess

# Các tham số bạn muốn truyền vào train.py
hf_token = "your_huggingface_token"
wandb_key = "your_wandb_key"
repo = "NV9523/DentalGPT_SFT"

# Gọi file train.py bằng subprocess
subprocess.run([
    "python", "Trainer/train.py",
    "--hf_token", hf_token,
    "--wandb_key", wandb_key,
    "--repo", repo
])
