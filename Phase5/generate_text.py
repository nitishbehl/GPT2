import sys, os
sys.path.append("/Users/nitishbehl/Desktop/gpt2_project/Phase1")  
from train_gpt2 import GPT, GPTConfig
import torch
import torch.nn.functional as F
import tiktoken


device = "mps" if torch.backends.mps.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")

def generate_text(prompt="Hello, I am a language model,", max_length=30):
    # Load model
    config = GPTConfig(
        block_size=1024,
        vocab_size=50257,
        n_layer=12,
        n_head=6,
        n_embd=384
    )
    model = GPT(config)
    model.load_state_dict(torch.load("/Users/nitishbehl/Desktop/gpt2_project/checkpoints/fine_tuned_gpt2_wikitext.pt", map_location=device), strict=False)
    model.to(device)
    model.eval()

    # Encode prompt
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Generate tokens
    sample_rng = torch.Generator(device=device).manual_seed(1337)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    decoded = enc.decode(x[0].tolist())
    return decoded
