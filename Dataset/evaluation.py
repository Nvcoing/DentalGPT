from unsloth import FastLanguageModel
import torch
from evaluate import load
import nltk
from bert_score import score as bert_score
from datasets import load_dataset

nltk.download('wordnet')

def load_model(model_name="NV9523/DentalGPT", max_seq_len=1024):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def format_prompt(i, q, g, r, j, a, f, c, s):
    base_prompt = (
        "<｜begin▁of▁sentence｜>"
        "<｜system｜>\n"
        f"### Hướng dẫn: {i.strip()}\n"
        "<｜user｜>\n"
        f"### Câu hỏi:\n {q.strip()}\n"
    )
    reference = (
        base_prompt +
        "<｜think｜>\n"
        "Hãy cùng diễn giải từng bước nào!🤔\n"
        "<reasoning_cot>\n"
        "# 🧠 Suy luận của DentalGPT\n"
        f"## 1️⃣ Mục tiêu 📌\n{g.strip()}\n"
        f"## 2️⃣ Bước suy nghĩ ⚙️\n{r.strip()}\n"
        f"## 3️⃣ Giải thích 📝\n{j.strip()}\n"
        "</reasoning_cot>\n"
        "<｜expert｜>\n"
        "<experting>\n"
        "# 👨‍🔬 Chuyên gia\n"
        f"## Trình bày dạng: {f.strip()}\n"
        f"## Nội dung về: {c.strip()}\n"
        f"## Chuyên sâu về: {s.strip()}\n"
        "</experting>\n"
        "<｜assistant｜>\n"
        "<answer>\n"
        f"# 💬 Câu trả lời\n{a.strip()}\n"
        "</answer>"
        "<｜end▁of▁sentence｜>"
    )
    return base_prompt, reference

def calculate_perplexity(model, tokenizer, texts):
    encodings = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to("cuda")
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()

def load_evaluate_dataset(model, tokenizer, dataset_name="NV9523/DentalGPT_SFT"):
    dataset = load_dataset(dataset_name, split="test")

    predictions = []
    references = []

    for ex in dataset:
        prompt, ref = format_prompt(
            ex["Instruction"], ex["Câu hỏi"],
            ex["CoT_Goal"], ex["CoT_Reasoning"],
            ex["CoT_Justification"], ex["Câu trả lời"],
            ex["label1"], ex["label2"], ex["label3"]
        )
        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_ids = model.generate(**input_ids, max_length=1024)
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        predictions.append(output.strip())
        references.append(ref.strip())

    print("=== Sample Output ===")
    for i in range(min(3, len(predictions))):  # Hiển thị 3 mẫu đầu
        print(f"\n--- Q{i+1} ---")
        print(f"Prediction:\n{predictions[i]}")
        print(f"Reference:\n{references[i]}")

    # Metrics
    bleu = load("bleu")
    rouge = load("rouge")
    meteor = load("meteor")

    bleu_result = bleu.compute(predictions=predictions, references=references)
    rouge_result = rouge.compute(predictions=predictions, references=references)
    meteor_result = meteor.compute(predictions=predictions, references=references)
    P, R, F1 = bert_score(predictions, references, lang="vi", verbose=False)
    ppl = calculate_perplexity(model, tokenizer, references)

    print("\n=== Evaluation Metrics ===")
    print(f"Perplexity: {ppl:.4f}")
    print(f"BLEU: {bleu_result['bleu']:.4f}")
    print(f"ROUGE: {rouge_result}")
    print(f"METEOR: {meteor_result['meteor']:.4f}")
    print(f"BERTScore F1: {F1.mean().item():.4f}")
