import os
import torch
import math
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from axiomatic_qfnn import AxiomaticQuantumField
from phaseview import phaseview_animation
# --- GPU and Precision Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BF16_SUPPORTED = torch.cuda.is_bf16_supported() if DEVICE == "cuda" else False

# --- Dataset and Model Config ---
DATASETS = {
    "wiki2": ("wikitext", "wikitext-2-raw-v1"),
    "wiki3": ("wikitext", "wikitext-103-raw-v1"),
    "c4": ("c4", "en"),
    "strawberry01": ("strawberry", "default")
}

DEFAULT_MODEL = "gpt2"
DEFAULT_OUTPUT_DIR = "./trained_model"

# --- Initialize Tokenizer ---
def init_model(model_name, precision="bf16"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# --- Load Dataset ---
def load_data(dataset_name, tokenizer):
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Available: {list(DATASETS.keys())}")
    name, subset = DATASETS[dataset_name]
    dataset = load_dataset(name, subset)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized

# --- Quantum Field Trainer with Early Stopping ---
def train_qfnn_on_dataset(dataset_name, model_name=DEFAULT_MODEL, output_dir=DEFAULT_OUTPUT_DIR,
                          precision="bf16", epochs=5, coherence_threshold=0.99, patience=3, top_n_save=5):
    tokenizer = init_model(model_name, precision)
    tokenized_datasets = load_data(dataset_name, tokenizer)

    model = AxiomaticQuantumField()
    history_all = {"loss": [], "phase_loss": [], "entropy": [], "coherence": []}
    best_samples = []

    os.makedirs(output_dir, exist_ok=True)

    patience_counter = 0

    for batch_idx, batch in enumerate(tokenized_datasets["train"]):
        text = tokenizer.decode(batch["input_ids"], skip_special_tokens=True)
        if not text.strip():
            print(f"⚠️ Skipping short input (0 tokens): '{text}'...")
            continue

        field, history = model.evolve(text)

        coherence = history["coherence"][-1]
        history_all["loss"].extend(history["loss"])
        history_all["phase_loss"].extend(history["phase_loss"])
        history_all["entropy"].extend(history["entropy"])
        history_all["coherence"].extend(history["coherence"])

        # Save best N samples
        best_samples.append((coherence, text))
        best_samples = sorted(best_samples, key=lambda x: x[0], reverse=True)[:top_n_save]

        if coherence >= coherence_threshold:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping: coherence {coherence:.4f} exceeded threshold for {patience} steps.")
                break
        else:
            patience_counter = 0

        if batch_idx % 10 == 0:
            print(f"✅ Batch {batch_idx} processed.")

        if batch_idx >= epochs:
            break

    # Save full training history
    torch.save(history_all, os.path.join(output_dir, "training_history.pt"))

    # Save top-N best samples
    with open(os.path.join(output_dir, "top_samples.txt"), "w", encoding="utf-8") as f:
        for coherence_score, sample_text in best_samples:
            f.write(f"Coherence: {coherence_score:.4f}\n{sample_text}\n\n")

    print(f"✅ Full QFNN Training Completed. History and top samples saved at {output_dir}")

    visualize_training(history_all)

# --- Visualization ---
def visualize_training(history):
    steps = range(len(history["loss"]))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax[0, 0].plot(steps, history["loss"])
    ax[0, 0].set_title("Total Loss")

    ax[0, 1].plot(steps, history["phase_loss"])
    ax[0, 1].set_title("Phase Loss")

    ax[1, 0].plot(steps, history["entropy"])
    ax[1, 0].set_title("Entropy")

    ax[1, 1].plot(steps, history["coherence"])
    ax[1, 1].set_title("Coherence")

    for a in ax.flat:
        a.set_xlabel("Steps")
        a.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./training_progress.png")
    print("🖼️ Training Progress plot saved at ./training_progress.png")
    plt.close()

# --- Main CLI Entry ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantum Field Trainer (QFNN Optimized)")
    parser.add_argument("--dataset", type=str, default="wiki2", choices=DATASETS.keys(), help="Dataset name")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="Tokenizer model name")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Where to save model & plots")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Floating point precision mode")
    parser.add_argument("--epochs", type=int, default=5, help="Number of batches to train")
    args = parser.parse_args()

    train_qfnn_on_dataset(
        args.dataset,
        args.model_name,
        args.output_dir,
        args.precision,
        args.epochs
    )
