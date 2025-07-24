
# imports and configuration setup
from dataclasses import dataclass
import math
import time
import torch
import os
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import tiktoken  # GPT-2 tokenizer library

# Step 2: Model Components
class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 128

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weightgit commit -m "first commit"git commit -m "first commit"s(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# Step 3: DataLoader
class DataLoaderLite:
    def __init__(self, B, T, process_rank=0, num_processes=1, split="train"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}
        self.data_root = "/Volumes/ExternalSSD/tokenized_cache"
        self.shards = sorted(f for f in os.listdir(self.data_root) if f.startswith(f"eduFineweb_{split}"))
        assert len(self.shards) > 0
        self.shards = [os.path.join(self.data_root, f) for f in self.shards]
        self.split = split
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.epoch_batches = 20
        self.max_tokens = self.B * self.T * self.epoch_batches + 1
        if len(self.tokens) < self.max_tokens:
            raise ValueError(f"Not enough tokens")
        self.current_position = self.B * self.T * self.process_rank

    def load_tokens(self, path):
        with open(path, "rb") as f:
            return torch.from_numpy(np.frombuffer(f.read(), dtype=np.uint16).copy()).to(torch.long)


    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + B * T * self.num_processes + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# Step 4: Optimizer

def configure_optimizers(model, weight_decay, learning_rate, device):
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

# Step 5: Training Loop

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(1337)
B = 16
T = 256
total_batch_size = 524288
grad_accum_steps = 2
train_loader = DataLoaderLite(B, T)
#val_loader = DataLoaderLite(B, T, split="val")
val_loader = None  # No validation data for now

model = GPT(GPTConfig()).to(device)
optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=6e-4, device=device)


# Setup log file
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f:
    pass


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 10

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

for step in range(max_steps):
    t0 = time.time()

    # if step % 100 == 0:
    #     model.eval()
    #     val_loader.reset()
    #     with torch.no_grad():
    #         val_loss_accum = 0.0
    #         for _ in range(20):
    #             x, y = val_loader.next_batch()
    #             x, y = x.to(device), y.to(device)
    #             _, loss = model(x, y)
    #             val_loss_accum += loss.detach()
    #         val_loss_accum /= 20
    #         print(f"[step {step}] validation loss: {val_loss_accum.item():.4f}")

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    dt = time.time() - t0
    tokens_processed = B * T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step:2d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss_accum.item():.6f}\n")


    import tiktoken

model.eval()
num_return_sequences = 4
max_length = 32
ddp_rank = 0

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
xgen = tokens.to(device)

sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(42 + ddp_rank)

while xgen.size(1) < max_length:
    with torch.no_grad():
        logits, loss = model(xgen)  # (B, T, vocab_size)
        logits = logits[:, -1, :]   # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        xgen = torch.cat((xgen, xcol), dim=1)

# Print the generated text
for i in range(num_return_sequences):
    tokens = xgen[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"rank {ddp_rank} sample {i}: {decoded}")
