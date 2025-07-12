from unsloth import FastLanguageModel
from transformers import AutoTokenizer

def load_model_and_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="NV9523/DentalGPT",
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
        torch_compile=False
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer
