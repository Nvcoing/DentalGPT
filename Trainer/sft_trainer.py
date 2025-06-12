import os
import wandb
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

def get_trainer(model, tokenizer, train_dataset, eval_dataset, repo_id, token, wandb_key=None):
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
        os.environ["WANDB_MODE"] = "online"
        
        wandb.init(
            project="DentalGPT",
            name=f"{repo_id.split('/')[-1]}",
            settings=wandb.Settings(
                save_code=False,         # không log code project
                _disable_stats=True      # không log CPU, GPU, RAM,...
            )
        )
        report_to = "wandb"
    else:
        report_to = "none"

    args = TrainingArguments(
        output_dir="DentalGPT_SFT",
        per_device_train_batch_size=4*4,
        gradient_accumulation_steps=2*4,
        warmup_steps=250,
        max_steps=None,
        learning_rate=2e-4, #2e-4 với 1k step rồi 5e-4 với 1k step 7e-4 với 1k step rồi 1e-3 đến hết (có thể để mặc định 5e-4)
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=int(100 / (4 * 4)),
        eval_steps=int(100 / (4 * 4)) if eval_dataset else None,
        save_strategy="steps",
        save_steps=int(1600 / (4 * 4)),
        save_total_limit=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to=report_to,
        dataloader_num_workers=12,
        save_on_each_node=False,
        logging_dir=None  # không lưu log tensorboard vào ổ đĩa
    )

    # Tính lại max_steps và logging/save_steps 
    epochs = 2
    steps = int(len(train_dataset) * epochs / (args.per_device_train_batch_size * args.gradient_accumulation_steps)) #189434 
    # steps = 189434  
    # args.warmup_steps = int(steps * 0.3)
    batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps #32
    batch_size = 32
    args.max_steps = steps
    args.logging_steps = int(800 / batch_size) #25
    args.save_steps = int(1600 / batch_size) #50

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        packing=False,
        args=args,
        dataset_num_proc=4,
        callbacks=[CheckpointPush(repo_id, token, args.save_steps)]
    )
    return trainer
