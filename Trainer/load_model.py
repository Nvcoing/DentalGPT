import torch
from unsloth import FastLanguageModel, is_bfloat16_supported

# Cấu hình chung
MAX_SEQ = 512
dTYPE = None  # auto detect
LOAD_4BIT = True
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Định nghĩa token đặc biệt
special_tokens = [
    "<|Paitent|>", "<|Goal|>", "<|Step_Reasoning|>",
    "<|Explain|>", "<|DentalGPT|>", "<|Question|>",
    "<|Think|>", "<|Answer|>"
]

# Tải model và tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ,
    dtype=dTYPE,
    load_in_4bit=LOAD_4BIT
)

# Thêm token đặc biệt vào tokenizer
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

# Resize embedding cho model (bắt buộc sau khi thêm token)
model.resize_token_embeddings(len(tokenizer))

# Thiết lập LoRA
PEFT_MODEL = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=128,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth"
)

# In thông tin tham số
trainable = sum(p.numel() for p in PEFT_MODEL.parameters() if p.requires_grad)
total = sum(p.numel() for p in PEFT_MODEL.parameters())
print(f"trainable params: {trainable}")
print(f"total params: {total}")
print(f"percent  trainable: {100*trainable/total:.4f}%")

# Expose
model = PEFT_MODEL
