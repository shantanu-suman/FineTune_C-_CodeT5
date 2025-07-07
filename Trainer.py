import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_from_disk

# ✅ Ensure GPU is available
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Please install PyTorch with GPU support.")

# ✅ Load full tokenized dataset
dataset_path = "D:/Python/DL/tokenized_juliet_dataset"
dataset = load_from_disk(dataset_path)

# ✅ Model + tokenizer
model_checkpoint = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, use_safetensors=True)

# ✅ Collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# ✅ Training settings (Adjusted for 8GB GPU)
batch_size = 4  # ✅ Reduce to fit in 8GB
gradient_accumulation_steps = 2  # Effective batch = 8
epochs = 3
total_examples = len(dataset)
steps_per_epoch = (total_examples // (batch_size * gradient_accumulation_steps)) + 1
total_training_steps = steps_per_epoch * epochs

# ✅ Save every 15% of total training
save_steps = max(1, int(total_training_steps * 0.05))

# ✅ Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="D:/Python/DL/codet5_explainer_checkpoints",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=epochs,
    learning_rate=5e-5,
    warmup_steps=200,
    optim="adamw_torch_fused",
    save_strategy="steps",
    save_steps=save_steps,
    logging_dir="D:/Python/DL/logs",
    logging_steps=100,
    eval_strategy="no",
    fp16=True,
    bf16=False,
    dataloader_pin_memory=True,
    dataloader_num_workers=4,  # ✅ Lower if CPU is under load
    report_to="none",
    save_total_limit=3,
    load_best_model_at_end=False,
    predict_with_generate=True
)

# ✅ Trainer setup
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ✅ Run training
if __name__ == "__main__":
    print("🚀 Training started...")
    trainer.train()
    print("✅ Training complete. Saving final model...")
    trainer.save_model("D:/Python/DL/codet5_explainer_model")
    print("📦 Final model saved to D:/Python/DL/codet5_explainer_model")
