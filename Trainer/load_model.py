import torch
from unsloth import FastLanguageModel, is_bfloat16_supported

MAX_SEQ = 2048
dTYPE = None  # auto detect
LOAD_4BIT = True
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ,
    dtype=dTYPE,
    load_in_4bit=LOAD_4BIT
)

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

trainable = sum(p.numel() for p in PEFT_MODEL.parameters() if p.requires_grad)
total = sum(p.numel() for p in PEFT_MODEL.parameters())
print(f"Trainable params: {trainable} | Total params: {total} | Percentage: {100*trainable/total:.2f}%")

model = PEFT_MODEL