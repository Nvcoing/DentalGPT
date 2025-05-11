from unsloth import FastLanguageModel
import torch

def load_model(model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
               max_seq_length: int = 512,
               dtype=None,
               load_in_4bit: bool = True):
    """
    Load pretrained model and tokenizer with given settings.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )

    # Gắn LoRA Adapter tại đây
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=128,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    return model, tokenizer
