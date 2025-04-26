
# 🌌 Quantum Field Neural Network (QFNN) — V3

---

## 🧠 Overview

This repository implements a **Quantum Field Neural Network** (QFNN) based on a **physics-grounded axiomatic framework**, upgraded with:

- 🌀 Polar Token Bank (dynamic phase embeddings)
- 🛠 Adaptive Field Diffusion (loss-driven exploration)
- 🌡 Thermodynamic Annealing (energy-based optimization)
- 🧬 Quantum Field Generation (symbolic dreaming)

It merges principles from:

- Quantum field theory
- Thermodynamics
- Information theory
- Symbolic sequence modeling

to evolve **complex-valued neural fields** for token learning and generation.

---

## 📂 System Modules

| File | Purpose |
| :--- | :------ |
| `axiomatic_qfnn_v3.py` | Core QFNN engine: field initialization, evolution, adaptive diffusion. |
| `trainer_qfnn_v3_patched.py` | Dataset loading, training orchestration, dynamic annealing. |
| `qfnn_v3_dynamic_generator_fixed.py` | Symbolic field dreaming engine — inference/generation module. |
| `polar_token_bank.pt` | Saved token polar coordinates for phase-based decoding. |
| `lab.ipynb` | Experimental notebook for field validation and dynamic generation. |

---

## ⚡ Physics Behind the Model

The **QFNN v3** is governed by **11 Enhanced Axioms**:

1. **Wave-Token Correspondence**  
2. **Golden Ratio Polar Initialization**  
3. **Energy-Level Radius Scaling**  
4. **Phase Alignment Objective**  
5. **Information Potential Field**  
6. **Field Diffusion Adaptation**  
7. **Simulated Annealing Temperature Control**  
8. **Hamiltonian-Loss Equivalence**  
9. **Radial Action Minimization**  
10. **Polar Memory Reweighting**  
11. **Quantum Path Integral Generation**

---

## 🛠️ How It Works

- Complex polar embeddings for tokens
- Heun-Euler integrator with stochastic diffusion
- Annealing schedule for loss minimization
- Hebbian field memory
- Dynamic symbolic dreaming

---

## 🚀 Running Training

```bash
python src/src2/trainer_qfnn_v3_patched.py --dataset wiki2 --epoch 5
```

---

## 🌠 Running Dreaming (Inference)

```python
from qfnn_v3_dynamic_generator_fixed import QFNNFieldExpanderV3

expander = QFNNFieldExpanderV3()
prompt = ["truth", "emerges", "from", "the", "hidden"]

dream = expander.generate_sequence(prompt, num_new_tokens=30)
print("🌌 Dream:", dream)
```

---

## 🧪 Scientific Validation Tools

- Phase lock plots
- Energy annealing curves
- Polar phase diffusion charts
- Sequence entropy trajectories

---

🌌 **QFNN is no longer just a model — it's a living symbolic field explorer.**  
🌌 **We don't predict tokens — we evolve meaning.**
