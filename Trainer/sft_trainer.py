# sft_trainer.py
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

def get_trainer(model, tokenizer, train_dataset,
                epoch: int = 20,
                per_device_train_batch_size: int = 4,
                gradient_accumulation_steps: int = 2,
                max_seq_length: int = 512,
                save_steps: int = 100,
                output_dir: str = "DentalGPT_SFT",
                hub_model_id: str = "NV9523/DentalGPT_SFT",
                hub_token: str = None):
    """
    Configure and return SFTTrainer instance.
    """
    total_steps = int(len(train_dataset) * epoch /
                      (per_device_train_batch_size * gradient_accumulation_steps))

    args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=250,
        max_steps=total_steps,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=100,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=output_dir,
        push_to_hub=True,
        hub_model_id=hub_model_id,
        hub_token=hub_token,
        report_to="none",
        dataloader_num_workers=2
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,
        packing=True,
        args=args
    )
    return trainer