from Dataset.evaluation import load_model, load_evaluate_dataset

if __name__ == "__main__":
    model, tokenizer = load_model()
    load_evaluate_dataset(model, tokenizer)
