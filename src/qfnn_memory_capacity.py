
import torch
import numpy as np
import matplotlib.pyplot as plt
from axiomatic_qfnn import AxiomaticQuantumField
from nltk.tokenize import word_tokenize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Metric: Memory Field Capacity Analyzer ---
def analyze_memory_capacity(text, model=None, steps=500, visualize=True):
    if model is None:
        model = AxiomaticQuantumField()

    tokens = model.tokenize(text)
    if len(tokens) < 3:
        print(f"⚠️ Text too short for analysis: {text}")
        return None

    field = model.init_field(tokens)
    θ_target = torch.roll(field["θ"], shifts=-1)

    for step in range(steps):
        field, loss, phase_loss, entropy, coherence = model.heun_step(field, θ_target, step)
    
    W = field["W"].cpu()
    W_abs = torch.abs(W)

    # Calculate Memory Field Metrics
    mean_strength = W_abs.mean().item()
    max_strength = W_abs.max().item()
    sparsity = (W_abs < 1e-3).float().mean().item()

    entropy_measure = -(W_abs/W_abs.sum()).log().mean().item()

    if visualize:
        plt.figure(figsize=(8, 6))
        plt.imshow(W_abs.numpy(), cmap="viridis")
        plt.colorbar(label="|W_ij|")
        plt.title("Hebbian Memory Field |W| after Evolution")
        plt.xlabel("Token j")
        plt.ylabel("Token i")
        plt.tight_layout()
        plt.show()

    print(f"✅ Memory Capacity Analysis:")
    print(f"• Mean Connection Strength: {mean_strength:.6f}")
    print(f"• Max Connection Strength: {max_strength:.6f}")
    print(f"• Sparsity (% near-zero): {sparsity*100:.2f}%")
    print(f"• Entropy of W Field: {entropy_measure:.6f}")

    return {
        "mean_strength": mean_strength,
        "max_strength": max_strength,
        "sparsity": sparsity,
        "entropy_W": entropy_measure
    }

# --- Example Usage ---
if __name__ == "__main__":
    text = "the mind stretches as light fills the infinite cosmos"
    analyze_memory_capacity(text)
