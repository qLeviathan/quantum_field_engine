
import torch
import numpy as np
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Build a token fieldbank: map vocab tokens into phase-space ---
def build_token_fieldbank(model_name="gpt2", max_tokens=5000):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = list(tokenizer.get_vocab().keys())[:max_tokens]

    # Phase embedding strategy: simple angle mapping
    N = len(vocab)
    golden_ratio = (1 + 5 ** 0.5) / 2
    i = torch.arange(N, dtype=torch.float32, device=DEVICE)
    θ = (i * golden_ratio * 2 * torch.pi) % (2 * torch.pi)

    r = 0.3 + 0.7 * torch.linspace(0, 1, N, device=DEVICE)
    real_part = r * torch.cos(θ)
    imag_part = r * torch.sin(θ)

    ψ_components = torch.stack([real_part, imag_part], dim=1)
    ψ_vocab = torch.view_as_complex(ψ_components)

    token_bank = {token: ψ_vocab[idx] for idx, token in enumerate(vocab)}
    return token_bank, tokenizer

# --- Example Usage ---
if __name__ == "__main__":
    bank, tokenizer = build_token_fieldbank()
    print(f"✅ Token Fieldbank built with {len(bank)} tokens.")
