import os
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
from huggingface_hub import upload_folder
from unsloth import is_bfloat16_supported
import wandb

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


def get_trainer(model, tokenizer, train_dataset, eval_dataset, repo_id, token, wandb_key=None):
    # Initialize WANDB nếu có key
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
        wandb.init(project="DentalGPT", name=f"{repo_id.split('/')[-1]}")
        report_to = "wandb"
    else:
        report_to = "none"
    
    args = TrainingArguments(
        output_dir="DentalGPT_SFT",
        per_device_train_batch_size=4*4,
        gradient_accumulation_steps=2*4,
        warmup_steps=250,
        max_steps=None,  # tính sau
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=int(100 / (4 * 4)),
        eval_steps=100/(4*4) if eval_dataset else None,
        save_strategy="steps",
        save_steps=int(200/(4*4)),
        save_total_limit=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to=report_to,
        dataloader_num_workers=4
    )
    
    # tính max_steps dựa vào dataset và epochs
    epochs = 2
    steps = int(len(train_dataset) * epochs / (args.per_device_train_batch_size * args.gradient_accumulation_steps))
    batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    args.max_steps = steps
    args.logging_steps = int(800 / batch_size)
    args.save_steps = int(1600 / batch_size)
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
        callbacks=[CheckpointPush(repo_id, token, args.save_steps)]
    )
    return trainer