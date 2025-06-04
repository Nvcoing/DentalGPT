from unsloth import FastLanguageModel
# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B for DentalGPTsmall
# deepseek-ai/DeepSeek-R1-Distill-Llama-8B for DentalGPTmedium
# deepseek-ai/DeepSeek-R1-Distill-Qwen-7B for DentalGPTlarge
# Cấu hình chung
MAX_SEQ = 1024
dTYPE = None  # auto detect
LOAD_4BIT = True
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Tải model và tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ,
    dtype=dTYPE,
    load_in_4bit=LOAD_4BIT
)

# Thiết lập LoRA
PEFT_MODEL = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=64,
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