# sft_trainer.py
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from unsloth import is_bfloat16_supported
from huggingface_hub import HfApi
import os

class UploadCheckpointCallback(TrainerCallback):
    def __init__(self, checkpoint_dir: str, repo_id: str, token: str):
        self.checkpoint_dir = checkpoint_dir
        self.repo_id = repo_id
        self.token = token
        self.api = HfApi()

    def on_save(self, args, state, control, **kwargs):
        print(f"Uploading checkpoint from {self.checkpoint_dir} to {self.repo_id}")
        self.api.upload_folder(
            repo_id=self.repo_id,
            folder_path=self.checkpoint_dir,
            token=self.token,
            repo_type="model",
            path_in_repo="checkpoint",
        )
        return control

def get_trainer(model, tokenizer, train_dataset,
                epoch: int = 20,
                per_device_train_batch_size: int = 4,
                gradient_accumulation_steps: int = 2,
                max_seq_length: int = 512,
                save_steps: int = 100,
                output_dir: str = "DentalGPT_SFT",
                hub_model_id: str = "NV9523/DentalGPT_SFT",
                hub_token: str = None):

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
        push_to_hub=False,  # <-- turn off auto push
        report_to="none",
        dataloader_num_workers=2
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True,
        args=args,
        callbacks=[
            UploadCheckpointCallback(
                checkpoint_dir=os.path.join(output_dir, f"checkpoint-{save_steps}"),
                repo_id=hub_model_id,
                token=hub_token
            )
        ]
    )
    return trainer
