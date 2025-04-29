from unsloth import FastLanguageModel
import torch

# Cấu hình tải mô hình
max_seq_length = 512  # Giới hạn tối đa 1024 tokens
dtype = None  # Auto detect Float16 hoặc BFloat16
load_in_4bit = True  # Sử dụng 4-bit quantization để tiết kiệm RAM
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit
)

# Cấu hình LoRA để fine-tune mô hình
model = FastLanguageModel.get_peft_model(
    model,
    r=4,  # Rank LoRA (càng cao càng tốn RAM, 16 là hợp lý)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=8,
    lora_dropout=0.05,  # Dropout tránh overfitting
    bias="none",
    use_gradient_checkpointing="unsloth",
)
