
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Setup Automatic Mixed Precision (AMP) for 4090 Speed ---
class GPUAccelerator:
    def __init__(self, use_amp=True):
        self.device = DEVICE
        self.use_amp = use_amp and DEVICE == "cuda"
        if self.use_amp:
            print("✅ AMP (mixed precision) enabled for maximum GPU speed.")
        else:
            print("ℹ️ AMP disabled or not available, using fp32.")

    def autocast(self):
        if self.use_amp:
            return torch.cuda.amp.autocast()
        else:
            return torch.no_grad()  # fallback if not AMP capable

    def scaler(self):
        if self.use_amp:
            return torch.cuda.amp.GradScaler()
        else:
            return None

# --- Example Usage ---
if __name__ == "__main__":
    accelerator = GPUAccelerator()
    with accelerator.autocast():
        x = torch.randn((4096, 4096), device=DEVICE)
        y = torch.matmul(x, x)
        print(f"✅ Matmul completed with shape {y.shape}")
