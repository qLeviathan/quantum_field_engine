from transformers import AutoTokenizer
import torch
import math

class QFNNPolarTokenBank:
    def __init__(self, model_name="gpt2", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokens = list(self.tokenizer.get_vocab().keys())
        self.golden_ratio = (1 + math.sqrt(5)) / 2
        self.token_polar_map = self.build_polar_token_map()
    
    def build_polar_token_map(self):
        N = len(self.tokens)
        i = torch.arange(N, dtype=torch.float32, device=self.device)
        
        # Phase angle with golden ratio spread
        θ = (i * self.golden_ratio * 2 * torch.pi) % (2 * torch.pi)
        
        # Radial position based on natural square-root filling
        r = 0.3 + 0.7 * torch.sqrt((i + 1.0) / N)

        token_polar_map = {}
        for idx, token in enumerate(self.tokens):
            token_polar_map[token] = (r[idx].item(), θ[idx].item())
        
        return token_polar_map
    
    def get_token_coordinates(self, token):
        return self.token_polar_map.get(token, None)
    
    def all_tokens(self):
        return self.token_polar_map