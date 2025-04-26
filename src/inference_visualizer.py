
import torch
import numpy as np
import matplotlib.pyplot as plt
from axiomatic_qfnn import AxiomaticQuantumField
from nltk.tokenize import word_tokenize
import pandas as pd

# 🔧 Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT = "the quantum field aligned each symbolic token like a spiral galaxy of meaning"

# 🔍 Initialize model
model = AxiomaticQuantumField()
tokens = model.tokenize(TEXT)
field = model.init_field(tokens)
θ_target = torch.roll(field["θ"], shifts=-1)

# 🧬 Run full field evolution
for step in range(model.config["steps"]):
    field, loss, phase_loss, entropy, coherence = model.heun_step(field, θ_target, step)

# 🎥 PhaseView Animation
θ_history = [torch.angle(field["ψ"]).cpu().numpy()]
r = torch.abs(field["ψ"]).cpu().numpy()
token_idx = np.arange(len(tokens))

plt.figure(figsize=(10, 6))
plt.polar(θ_history[0], r, marker='o')
plt.title("Final Token Phase Field")
plt.savefig("phaseview_inference.png")
plt.close()

# 🔥 Memory Field Visualization
W = field["W"].cpu().abs().numpy()
plt.figure(figsize=(8, 6))
plt.imshow(W, cmap='inferno')
plt.colorbar(label="|W_ij|")
plt.title("Hebbian Memory Field (|W|)")
plt.xlabel("Token j")
plt.ylabel("Token i")
plt.savefig("memory_W_heatmap.png")
plt.close()

# 📋 Token Table
ψ_final = field["ψ"].cpu()
θ = torch.angle(ψ_final).numpy()
r = torch.abs(ψ_final).numpy()
target = torch.roll(torch.tensor(θ), shifts=-1).numpy()
delta = np.abs((θ - target + np.pi) % (2 * np.pi) - np.pi)

df = pd.DataFrame({
    "Token": tokens,
    "θ (rad)": θ,
    "r (magnitude)": r,
    "θ_target": target,
    "Δθ (error)": delta
})
df.to_csv("token_field_summary.csv", index=False)

print("✅ Inference complete. Outputs:")
print("- phaseview_inference.png")
print("- memory_W_heatmap.png")
print("- token_field_summary.csv")
