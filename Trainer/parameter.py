from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import basemodel as LoadModel
import load as load_dataset
model, tokenizer = LoadModel()
dataset = load_dataset("./Build dataset Dental/Process_data/Data_processed/Dental_CoT_dataset.xlsx")
def Trainer(
    model,
    tokenizer,
    dataset,
    max_seq_length=512,
    output_dir="./Model/dentalGPT",
    per_device_train_batch_size=10,
    gradient_accumulation_steps=3,
    warmup_steps=10,
    max_steps=1000,
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    lr_scheduler_type="linear",
    seed=42,
    dataloader_num_workers=4,
    dataset_text_field="text",
    dataset_num_proc=1,
    packing=False,
    report_to="none"
):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=dataset_text_field,
        max_seq_length=max_seq_length,
        dataset_num_proc=dataset_num_proc,
        packing=packing,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            optim="adamw_8bit",
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            seed=seed,
            output_dir=output_dir,
            report_to=report_to,
            dataloader_num_workers=dataloader_num_workers
        ),
    )
    return trainer
trainer = Trainer(model, tokenizer, dataset)
