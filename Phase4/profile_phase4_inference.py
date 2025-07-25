import torch
import time
import sys
sys.path.append('../Phase2')
from train_phase2_wikitext import GPT, GPTConfig
import tiktoken
from datasets import load_dataset

# === Config ===
device = "mps" if torch.backends.mps.is_available() else "cpu"

# === Load model ===
config = GPTConfig(n_layer=12)
model = GPT(config).to(device)
model.load_state_dict(torch.load("checkpoints/fine_tuned_gpt2_wikitext.pt", map_location=device))
model.eval()

# === Tokenize sample ===
enc = tiktoken.get_encoding("gpt2")
text = "The quick brown fox jumps over the lazy dog."
tokens = enc.encode(text)
x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # (1, T)

# === Profiling ===
print(f"{'Layer':>5} | {'Time (ms)':>10} | {'Output shape':>15}")
print("-" * 40)

with torch.no_grad():
    B, T = x.shape
    pos = torch.arange(0, T, device=device)
    hidden = model.transformer.wte(x) + model.transformer.wpe(pos)

    for i, block in enumerate(model.transformer.h):
        start_time = time.time()
        hidden = block(hidden)
        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000
        print(f"{i+1:>5} | {elapsed_ms:>10.2f} | {str(tuple(hidden.shape)):>15}")

# === Total final layer ===
hidden = model.transformer.ln_f(hidden)
logits = model.lm_head(hidden)
print(f"\nFinal logits shape: {logits.shape}")
