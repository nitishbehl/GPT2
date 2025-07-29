import sys
import time
import json
import torch
import tiktoken

sys.path.append("../Phase2")
from train_phase2_wikitext import GPT, GPTConfig
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
x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # shape: (1, T)

# === Initialize hidden ===
B, T = x.shape
pos = torch.arange(0, T, device=device)
hidden = model.transformer.wte(x) + model.transformer.wpe(pos)

# === Profiling ===
print(f"{'Layer':>5} | {'Time (ms)':>10} | {'Memory (MB)':>12} | {'Output shape':>15}")
print("*" * 60)

profile_data = []

for i, block in enumerate(model.transformer.h):
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    before_mem = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    start_time = time.time()

    hidden = block(hidden)

    elapsed_time = (time.time() - start_time) * 1000  # in ms
    after_mem = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    mem_used = (after_mem - before_mem) / 1e6 if torch.cuda.is_available() else 0.0

    print(f"{i:5d} | {elapsed_time:10.2f} | {mem_used:12.2f} | {str(tuple(hidden.shape)):>15}")
    profile_data.append({
        "layer": i,
        "time_ms": round(elapsed_time, 3),
        "memory_mb": round(mem_used, 3),
        "output_shape": list(hidden.shape)
    })

hidden = model.transformer.ln_f(hidden)
logits = model.lm_head(hidden)


print(f"\nFinal logits shape: {logits.shape}")

# === Save profiling ===
output_path = "Phase4/log/phase4_profile_finetuned.json"
with open(output_path, "w") as f:
    json.dump(profile_data, f, indent=4)

print(f"\n[Saved] Profiling data written to {output_path}")
