import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from axiomatic_qfnn_v3 import AxiomaticQuantumFieldV3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "gpt2"
DEFAULT_OUTPUT_DIR = "./trained_model_v3"

DATASETS = {
    "wiki2": ("wikitext", "wikitext-2-raw-v1"),
    "wiki3": ("wikitext", "wikitext-103-raw-v1"),
    "c4": ("c4", "en"),
    "strawberry01": ("strawberry", "default")
}

def init_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_data(dataset_name, tokenizer):
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not found in list.")
    name, subset = DATASETS[dataset_name]
    dataset = load_dataset(name, subset)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized

def train_qfnn_v3(dataset_name="wiki2", model_name=DEFAULT_MODEL, output_dir=DEFAULT_OUTPUT_DIR,
                  epochs=5, steps_per_text=500):
    tokenizer = init_tokenizer(model_name)
    tokenized_datasets = load_data(dataset_name, tokenizer)

    model = AxiomaticQuantumFieldV3()
    history_all = {"loss": [], "entropy": []}
    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, batch in enumerate(tokenized_datasets["train"]):
        text = tokenizer.decode(batch["input_ids"], skip_special_tokens=True)
        tokens = text.split()

        if len(tokens) < 3:
            continue

        field = model.init_field(tokens)
        θ_target = torch.roll(field["θ"], shifts=-1)

        for step in range(steps_per_text):
            field = model.vectorized_field_update(field, θ_target)
            log_loss, entropy = model.compute_loss(field["ψ"], θ_target)
            history_all["loss"].append(log_loss.item())
            history_all["entropy"].append(entropy.item())

            if step % 50 == 0:
                print(f"[{batch_idx}:{step}] Loss: {log_loss:.4f} | Entropy: {entropy:.4f}")

        if batch_idx >= epochs:
            break

    torch.save(history_all, os.path.join(output_dir, "training_history_v3.pt"))
    visualize_training(history_all, output_dir)

def visualize_training(history, output_dir):
    steps = range(len(history["loss"]))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(steps, history["loss"])
    ax[0].set_title("Log Loss")
    ax[0].set_xlabel("Steps")

    ax[1].plot(steps, history["entropy"])
    ax[1].set_title("Entropy")
    ax[1].set_xlabel("Steps")

    for a in ax:
        a.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress_v3.png"))
    plt.close()
    print(f"🖼️ Training Progress saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QFNN v3 Trainer")
    parser.add_argument("--dataset", type=str, default="wiki2", choices=DATASETS.keys(), help="Dataset name")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="Tokenizer model name")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Where to save outputs")
    parser.add_argument("--epochs", type=int, default=5, help="How many texts to train on")
    args = parser.parse_args()

    train_qfnn_v3(args.dataset, args.model_name, args.output_dir, args.epochs)