from trl import ORPOTrainer, ORPOConfig
from unsloth import is_bfloat16_supported

def create_orpo_trainer(model, tokenizer, dataset):
    config = ORPOConfig(
        remove_unused_columns=False,
        max_length=1024,
        max_prompt_length=256,
        max_completion_length=768,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        beta=0.2,
        max_steps=(len(dataset)*5)//8,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        learning_rate=1e-5,
        output_dir="DentalGPT_RLHF",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
        report_to="none",
    )

    trainer = ORPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=config,
    )
    return trainer
