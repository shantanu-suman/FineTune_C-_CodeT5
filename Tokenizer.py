import os
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


# ✅ Use a seq2seq-compatible tokenizer like CodeT5
model_checkpoint = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

input_dir = r"D:\Python\DL\parsed_juliet_codet5_blocks"
jsonl_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".jsonl")]

# Load all .jsonl files and concatenate into one dataset
datasets_list = [load_dataset("json", data_files=f)["train"] for f in jsonl_files]
full_dataset = concatenate_datasets(datasets_list)

print("Sample input:", full_dataset[0]["source"])
print("Sample output:", full_dataset[0]["explanation"])

# ✅ Tokenization function for seq2seq
def tokenize_seq2seq(example):
    model_inputs = tokenizer(
        example["source"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["explanation"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ✅ Apply tokenization
tokenized_dataset = full_dataset.map(tokenize_seq2seq, batched=True)

# Save for later training
tokenized_dataset.save_to_disk("tokenized_juliet_seq2seq")
print("Sample input:", full_dataset[0]["source"])
print("Sample output:", full_dataset[0]["explanation"])

print("✅ Tokenized dataset saved to 'tokenized_juliet_seq2seq'")
