# load_model.py
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
    return model, tokenizer