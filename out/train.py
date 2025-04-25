import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from qfnn 
# --- Configuration ---
DATASETS = {
    "wiki2": ("wikitext", "wikitext-2-raw-v1"),
    "wiki3": ("wikitext", "wikitext-103-raw-v1"),
    "c4": ("c4", "en"),
    "strawberry01": ("strawberry", "default")  # Custom dataset assumed to be in HuggingFace Datasets format
}

DEFAULT_MODEL = "gpt2"
DEFAULT_OUTPUT_DIR = "./trained_model"

# --- Tokenizer & Model Initialization ---
def init_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# --- Dataset Loader ---
def load_data(dataset_name, tokenizer):
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' is not supported. Choose from {list(DATASETS.keys())}")
    name, subset = DATASETS[dataset_name]
    dataset = load_dataset(name, subset)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized

# --- Training Routine ---
def train_model(dataset_name, model_name=DEFAULT_MODEL, output_dir=DEFAULT_OUTPUT_DIR):
    tokenizer, model = init_model(model_name)
    tokenized_datasets = load_data(dataset_name, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=2,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"✅ Model trained and saved to {output_dir}")

# --- Main Entry ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monolithic Trainer for HuggingFace Datasets")
    parser.add_argument("--dataset", type=str, default="wiki2", choices=DATASETS.keys(), help="Dataset name")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="Model checkpoint name")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for trained model")

    args = parser.parse_args()
    train_model(args.dataset, args.model_name, args.output_dir)