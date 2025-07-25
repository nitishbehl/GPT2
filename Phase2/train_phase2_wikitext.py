import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
from datasets import load_dataset
import tiktoken
import os

# ========= Configs =========
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 384

# ========= Model =========
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size), persistent=False)


    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None, return_block9=False):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)

        block9_out = None
        for i, block in enumerate(self.transformer.h):
            x = block(x)
            if i == 5:  # Capture output after 9th transformer block (index 8)
                block9_out = x.clone()
                
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        if return_block9:
            return logits, loss, block9_out
        else:
            return logits, loss


# ========= Tokenization =========
print(" Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
enc = tiktoken.get_encoding("gpt2")

print("  Tokenizing...")
all_tokens = []
for item in dataset:
    text = item["text"]
    if text.strip() == "":
        continue
    all_tokens.extend(enc.encode(text))
tokens = torch.tensor(all_tokens, dtype=torch.long)

# ========= Batching =========
B, T = 4, 256
num_batches = len(tokens) // (B * T)
tokens = tokens[:num_batches * B * T]
x_data = tokens.view(B, -1)  # (B, N)

def get_batch(step):
    start = step * T
    end = start + T
    if end + 1 >= x_data.size(1):
        return None, None
    x = x_data[:, start:end]
    y = x_data[:, start+1:end+1]
    return x, y

# ========= Train =========
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = GPT(GPTConfig()).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

steps = 200
grad_accum = 4

print(" Training begins...\n")
for step in range(steps):
    optimizer.zero_grad()
    total_loss = 0.0

    for _ in range(grad_accum):
        x, y = get_batch(step)
        if x is None: break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        loss = loss / grad_accum
        loss.backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    print(f"Step {step:03d} | Loss: {total_loss:.4f}")

# ========= Save =========
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/fine_tuned_gpt2_wikitext.pt")
print("\n Model saved to checkpoints/fine_tuned_gpt2_wikitext.pt")
