
import torch
import numpy as np
import torch.nn.functional as F
from axiomatic_qfnn import AxiomaticQuantumField
from nltk.tokenize import word_tokenize
from datasets import load_dataset
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Metric 1: Perplexity Calculator ---
def evaluate_perplexity_on_text(text, model=None, steps=500):
    if model is None:
        model = AxiomaticQuantumField()
        
    tokens = model.tokenize(text)
    if len(tokens) < 3:
        print(f"⚠️ Text too short for evaluation: {text}")
        return None

    field = model.init_field(tokens)
    θ_target = torch.roll(field["θ"], shifts=-1)

    loss_total = 0.0
    for step in range(steps):
        field, loss, phase_loss, entropy, coherence = model.heun_step(field, θ_target, step)
        loss_total += loss.item()
    
    avg_loss = loss_total / steps
    perplexity = math.exp(avg_loss)
    print(f"✅ Evaluated Perplexity: {perplexity:.2f}")
    return perplexity

# --- Metric 2: Symbolic Decoder ---
def decode_field(field, model=None):
    if model is None:
        model = AxiomaticQuantumField()

    ψ = field["ψ"].cpu()
    r = torch.abs(ψ)
    θ = torch.angle(ψ)
    
    # Sort tokens by angle
    angles_sorted_idx = torch.argsort(θ)
    
    # Get corresponding tokens
    tokens = model.tokenize("placeholder text to recover")
    tokens_sorted = [tokens[i] for i in angles_sorted_idx]
    
    print(f"🧠 Symbolic Decoded Token Order (by Phase): {tokens_sorted}")
    return tokens_sorted

# --- Example Usage ---
if __name__ == "__main__":
    model = AxiomaticQuantumField()
    sample_text = "understanding is light and truth flows across space"
    evaluate_perplexity_on_text(sample_text, model)
