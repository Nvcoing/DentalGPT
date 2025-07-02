import torch
from unsloth import FastLanguageModel

model_name = "NV9523/DentalGPT"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True
)
FastLanguageModel.for_inference(model)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
