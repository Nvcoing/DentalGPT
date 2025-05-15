import os
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
from huggingface_hub import upload_folder
from unsloth import is_bfloat16_supported

class CheckpointPush(TrainerCallback):
    def __init__(self, repo_id: str, token: str, save_steps: int):
        self.repo_id = repo_id
        self.token = token
        self.save_steps = save_steps

    def on_save(self, args, state, control, **kwargs):
        if state.is_local_process_zero and state.global_step % self.save_steps == 0:
            ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            upload_folder(
                folder_path=ckpt,
                repo_id=self.repo_id,
                token=self.token,
                path_in_repo=f"checkpoint-{state.global_step}"
            )
            print(f"pushed checkpoint-{state.global_step}")
        return control


def get_trainer(model, tokenizer, dataset, repo_id, token):
    args = TrainingArguments(
        output_dir="DentalGPT_SFT",
        per_device_train_batch_size=4*2,
        gradient_accumulation_steps=2*2,
        warmup_steps=250,
        max_steps=None,  # tính sau
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
        dataloader_num_workers=4
    )
    # tính max_steps dựa vào dataset và epochs
    epochs = 1
    steps = int(len(dataset) * epochs / (args.per_device_train_batch_size * args.gradient_accumulation_steps))
    args.max_steps = steps

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        packing=True,
        args=args,
        dataset_num_proc=2,
        callbacks=[CheckpointPush(repo_id, token, args.save_steps)]
    )
    return trainer