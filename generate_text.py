import torch
import torch.nn.functional as F
from train_gpt2 import GPT, GPTConfig
import tiktoken

device = "mps" if torch.backends.mps.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")

# Load the trained model
model = GPT(GPTConfig())
model.load_state_dict(torch.load("model.pt", map_location=device))  # Make sure model.pt exists
model.to(device)
model.eval()

prompt = "Hello, I am a language model,"
tokens = enc.encode(prompt)
x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

max_length = 30
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
print("Generated:", decoded)

