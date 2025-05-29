import os
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
from huggingface_hub import upload_folder
from unsloth import is_bfloat16_supported
import wandb

class CheckpointPush(TrainerCallback):
    def __init__(self, repo_id: str, hf_token: str, save_steps: int):
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.save_steps = save_steps

    def on_save(self, args, state, control, **kwargs):
        if state.is_local_process_zero and state.global_step % self.save_steps == 0:
            ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            upload_folder(
                folder_path=ckpt,
                repo_id=self.repo_id,
                token=self.hf_token,
                path_in_repo=f"checkpoint-{state.global_step}"
            )
            print(f"Pushed checkpoint-{state.global_step} to HuggingFace Hub")
        return control


def get_trainer(model, tokenizer, train_dataset, eval_dataset, repo_id, hf_token, wandb_key):
    # Initialize WANDB
    wandb.login(key=wandb_key)
    wandb.init(
        project="DentalGPT",
        name=repo_id.split('/')[-1],
        config={
            "model": repo_id,
            "learning_rate": 2e-4,
            "architecture": "LLM Fine-tuning",
            "dataset": "DentalGPT_SFT"
        }
    )
    
    args = TrainingArguments(
        output_dir="DentalGPT_SFT",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=50,
        max_steps=1000,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to="wandb",
        dataloader_num_workers=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=True,
        args=args,
        dataset_num_proc=2,
        callbacks=[CheckpointPush(repo_id, hf_token, args.save_steps)]
    )
    return trainer