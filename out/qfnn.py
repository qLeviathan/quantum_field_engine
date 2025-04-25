# quantum_field_engine_monolithic.py
import torch
import math
from nltk.tokenize import word_tokenize
from text_decoder import field_to_sentence, decode_from_field
import torch.nn.functional as F
import nltk
nltk.download("punkt")
nltk.download('averaged_perceptron_tagger_eng')
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

# --- CONFIG ---
Φ = (1 + math.sqrt(5)) / 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG = {
    "η_0": 0.002,
    "λ": 0.75,           # phase loss to learning rate sensitivity
    "μ": 0.2,            # memory modulation coefficient
    "hebb_lr": 0.05,
    "hebb_decay": 0.99,
    "entropy_weight": 0.01,
    "pareto_k": 0.25,
    "h_base": 0.01,
    "steps": 500,
    "epsilon": 1e-8,
}

# --- TOKENIZER ---
def tokenize(text):
    return word_tokenize(text.lower())

# --- INIT FIELD ---
def init_field(tokens):
    N = len(tokens)
    i = torch.arange(N, dtype=torch.float32, device=DEVICE)
    θ = (i * Φ * 2 * torch.pi) % (2 * torch.pi)
    r = 0.3 + 0.7 * torch.linspace(0, 1, N, device=DEVICE)
    ψ = torch.randn(N, 2, device=DEVICE)
    ψ = torch.view_as_complex(ψ)
    W = torch.zeros((N, N), dtype=torch.cfloat, device=DEVICE)
    return θ, r, ψ, W

# --- LOSS + ENTROPY ---
def compute_loss(ψ, θ_pred, θ_target):
    phase_loss = torch.mean((θ_pred - θ_target) ** 2)
    mag_sq = torch.abs(ψ) ** 2
    entropy = -torch.sum(mag_sq * torch.log(mag_sq + CONFIG["epsilon"]))
    return phase_loss, entropy

# --- PARETO MASK ---
def pareto_mask(metric, k):
    threshold = torch.quantile(metric, 1 - k)
    return metric >= threshold

# --- HEBBIAN UPDATE ---
def update_hebbian(W, ψ, η, decay):
    ΔW = η * torch.outer(ψ.conj(), ψ)
    return decay * W + ΔW

# --- SCHRÖDINGER PROPAGATION ---

def propagate(ψ, V, h):
    # First Euler step
    k1 = -ψ + V * ψ
    ψ_mid = ψ + h * k1
    # Normalize midpoint
    ψ_mid = F.normalize(ψ_mid, p=2, dim=0)  # unit-norm across field
    # Recalculate derivative at midpoint
    k2 = -ψ_mid + V * ψ_mid
    # Final Heun step
    ψ_next = ψ + 0.5 * h * (k1 + k2)
    # Final normalization (unit-energy)
    ψ_next = F.normalize(ψ_next, p=2, dim=0)

    return ψ_next

    
# --- Shock with cohernece approaching .99---

def apply_inverse_beta_shock(ψ, step, T, beta_min=0.1, beta_max=20.0, α=2.0, δ=0.1, Φ=(1 + math.sqrt(5)) / 2):
    β = beta_min + (beta_max - beta_min) * (step / T)**α * (1 + δ * math.sin(2 * math.pi * Φ * step / T))
    noise = (1.0 / β) * torch.randn_like(ψ)
    return ψ + noise

# --- COUPLED SYSTEM EVOLUTION ---
def evolve(text):
    tokens = tokenize(text)
    N = len(tokens)
    θ, r, ψ, W = init_field(tokens)
    F_stack = torch.zeros_like(ψ)
    θ_target = torch.roll(θ, shifts=-1)

    for step in range(CONFIG["steps"]):
        θ_pred = torch.angle(ψ)
        θ_error = torch.abs(θ_pred - θ_target)

        phase_loss = torch.mean(θ_error ** 2)
        mag_sq = torch.abs(ψ) ** 2
        entropy = -torch.sum(mag_sq * torch.log(mag_sq + CONFIG["epsilon"]))
        loss_total = phase_loss + CONFIG["entropy_weight"] * entropy

        # Vectorized Pareto top-k mask
        k = int(CONFIG["pareto_k"] * N)
        topk = torch.topk(θ_error, k=k)
        mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        mask[topk.indices] = True

        ψ_selected = ψ.clone()
        ψ_selected[~mask] = ψ.detach()[~mask]

        # Vectorized η update
        memory_pressure = torch.sum(torch.abs(W), dim=1)  # (N,)
        η = CONFIG["η_0"] + CONFIG["λ"] * phase_loss + CONFIG["μ"] * memory_pressure  # (N,)
        η = η.unsqueeze(1)  # shape: (N,1)

        # Compute potential + memory field stack
        V = (W @ ψ).real + CONFIG.get("stack_decay", 0.95) * F_stack.real
        ψ = propagate(ψ_selected, V, CONFIG["h_base"] * η.squeeze())  # η is per-token

        # Hebbian update (in-place)
        ΔW = CONFIG["hebb_lr"] * torch.outer(ψ.conj(), ψ)
        W.mul_(CONFIG["hebb_decay"]).add_(ΔW)

        # Update memory trace
        F_stack = CONFIG.get("stack_decay", 0.95) * F_stack + ψ

        # Logging
        if step % 50 == 0 or step == CONFIG["steps"] - 1:
            coherence = torch.abs(torch.mean(torch.exp(1j * θ_pred)))
            if coherence > 0.99:
                ψ = apply_inverse_beta_shock(ψ, step, CONFIG["steps"])
                print(f"⚡ Entropy shock injected at step {step} | β adjusted for phase exploration.")
            print(f"[{step}] Loss: {loss_total:.4f} | Phase: {phase_loss:.4f} | Entropy: {entropy:.4f} | Coherence: {coherence:.4f}")
            print(f"    η mean: {η.mean().item():.4f} | F_stack norm: {F_stack.norm().item():.4f}")


    return ψ, W, F_stack, θ_target

def compose_fields(fields, method='average'):
    assert len(fields) > 1, "Need at least two fields to compose."

    shapes = [f["ψ"].shape for f in fields]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError("All fields must have the same ψ dimensionality.")

    ψ_stack = torch.stack([f["ψ"].to(DEVICE) for f in fields])
    W_stack = torch.stack([f["W"].to(DEVICE) for f in fields])
    F_stack = torch.stack([f["F_stack"].to(DEVICE) for f in fields])

    if method == 'average':
        ψ_fused = ψ_stack.mean(dim=0)
        W_fused = W_stack.mean(dim=0)
        F_fused = F_stack.mean(dim=0)
    elif method == 'sum':
        ψ_fused = ψ_stack.sum(dim=0)
        W_fused = W_stack.sum(dim=0)
        F_fused = F_stack.sum(dim=0)
    elif method == 'max':
        ψ_fused = ψ_stack.max(dim=0).values
        W_fused = W_stack.max(dim=0).values
        F_fused = F_stack.max(dim=0).values
    else:
        raise ValueError("Unknown composition method.")

    # Decode the fused field
    θ_target = torch.roll(torch.angle(ψ_fused), shifts=-1)
    decoded_tokens = decode_from_field(torch.angle(ψ_fused), θ_target)
    decoded_sentence = field_to_sentence(decoded_tokens)

    return {
        "ψ": ψ_fused,
        "W": W_fused,
        "F_stack": F_fused,
        "decoded": decoded_tokens,
        "sentence": decoded_sentence
    }

def run_qfnn_on_corpus(text, window_size=32, stride=16):
    from nltk.tokenize import sent_tokenize

    checkpoints = []
    sentences = sent_tokenize(text)
    current = ""
    
    for sentence in sentences:
        if len(current.split()) < window_size:
            current += " " + sentence
        else:
            evolve_from_text(current.strip())
            checkpoint = torch.load("semantic_field_checkpoint.pt")
            checkpoints.append(checkpoint)
            current = sentence
    
    if current.strip():
        evolve_from_text(current.strip())
        checkpoint = torch.load("semantic_field_checkpoint.pt")
        checkpoints.append(checkpoint)
    
    return checkpoints

def save_checkpoint(path, ψ, W, F_stack, meta=None):
    torch.save({
        "ψ": ψ.detach().cpu(),
        "W": W.detach().cpu(),
        "F_stack": F_stack.detach().cpu(),
        "meta": meta or {}
    }, path)
    print(f"💾 Field memory saved: {path}")

def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    return checkpoint["ψ"].to(DEVICE), checkpoint["W"].to(DEVICE), checkpoint["F_stack"].to(DEVICE), checkpoint.get("meta", {})


import matplotlib.pyplot as plt

def visualize_field(ψ, title="Quantum Field ψ"):
    amp = torch.abs(ψ).cpu().numpy()
    phase = torch.angle(ψ).cpu().numpy()
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(amp)
    ax[0].set_title("Amplitude")
    ax[1].plot(phase)
    ax[1].set_title("Phase")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def decode_stream_from_ψ(ψ, θ_target):
    θ_fused = torch.angle(fused["ψ"])
    θ_target_fused = torch.roll(θ_fused, shifts=-1)
    decoded_tokens = decode_from_field(θ_fused, θ_target_fused)
    sentence = field_to_sentence(decoded_tokens)
    print("🧠 Fused Field Decoded:", sentence)

    return sentence

def train_from_txt_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return run_qfnn_on_corpus(text)

def evolve_from_text(text, save_path="semantic_field_checkpoint.pt"):
    ψ, W, F_stack, _ = evolve(text)
    save_checkpoint(save_path, ψ, W, F_stack)

# EXAMPLE: Compose 2 different field runs
evolve_from_text("a field of intelligence evolves")
cp1 = torch.load("semantic_field_checkpoint.pt")

evolve_from_text("memory and resonance organize coherence")
cp2 = torch.load("semantic_field_checkpoint.pt")

fused = compose_fields([cp1, cp2])
# Recalculate θ_target for the fused field
θ_target_fused = torch.roll(torch.angle(fused["ψ"]), shifts=-1)

# Decode the sentence
decoded_tokens = decode_from_field(torch.angle(fused["ψ"]), θ_target_fused)
sentence = field_to_sentence(decoded_tokens)

print("🧠 Fused Field Decoded:", sentence)


# EXAMPLE: Run across a large text
long_text = open("out\input.txt").read()
checkpoints = run_qfnn_on_corpus(long_text)


# --- RUN ---
if __name__ == "__main__":
    text = "consciousness emerges from coherent interference patterns in the semantic quantum field"
    ψ, W, F_stack, θ_target = evolve(text)
    # Save field memory
    
    save_checkpoint("coherent_memory.pt", ψ, W, F_stack)


    ψ, W, F_stack, meta = load_checkpoint("coherent_memory.pt")

    #Load & Decode It
    θ_target = torch.roll(torch.angle(ψ), shifts=-1)
    decode_stream_from_ψ(ψ, θ_target)
    visualize_field(ψ, title="ψ from 'coherent memory' concept")

    checkpoints = train_from_txt_file("out\input.txt")
    visualize_field(checkpoints[0]["ψ"], title="Field from Corpus Line 1")
