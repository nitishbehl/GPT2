import sys
import torch
import torch.nn.functional as F
import tiktoken
import time

# Add Phase1 and Phase3 directories to sys.path
sys.path.append("/Users/nitishbehl/Desktop/gpt2_project/Phase3")
from early_exit import EarlyExitClassifier

sys.path.append("/Users/nitishbehl/Desktop/gpt2_project/Phase1")
from train_gpt2 import GPT, GPTConfig

device = "mps" if torch.backends.mps.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")

# Load early exit classifier
early_exit_classifier = EarlyExitClassifier().to(device)
early_exit_classifier.load_state_dict(
    torch.load(
        "/Users/nitishbehl/Desktop/gpt2_project/Phase3/checkpoints/early_exit_classifier.pt",
        map_location=device,
    )
)
early_exit_classifier.eval()

MODEL_PATHS = {
    "finetuned": "/Users/nitishbehl/Desktop/gpt2_project/Phase2/checkpoints/fine_tuned_gpt2_wikitext.pt",
    "pretrained": "/Users/nitishbehl/Desktop/gpt2_project/Phase1/checkpoints/pretrained_gpt2.pt",
    "earlyexit": "/Users/nitishbehl/Desktop/gpt2_project/Phase3/checkpoints/fine_tuned_gpt2_wikitext.pt",
}

_loaded_models = {}

def load_model(model_name: str):
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    config = GPTConfig(
        block_size=1024,
        vocab_size=50257,
        n_layer=12,
        n_head=6,
        n_embd=384
    )
    model = GPT(config)

    checkpoint = MODEL_PATHS.get(model_name)
    if checkpoint:
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"[WARNING] No checkpoint for {model_name}, using random weights.")

    model.to(device)
    model.eval()
    _loaded_models[model_name] = model
    return model

def generate_text(prompt="Hello", model="finetuned", max_length=50):
    model = model.lower().strip()
    prompt = prompt.strip()

    if not prompt:
        return "[Error] Empty prompt. Please enter something."

    model_instance = load_model(model)
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            if model == "earlyexit":
                logits, all_hidden = model_instance(x, return_all_hidden=True)
                hidden9 = all_hidden[9]  # Shape: [1, seq_len, 384]
                last_hidden = hidden9[:, -1, :]  # Only take last token
                logits = early_exit_classifier(last_hidden)
            else:
                logits_out = model_instance(x)
                if isinstance(logits_out, tuple):
                    logits = logits_out[0]
                else:
                    logits = logits_out
                logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)

    output_tokens = x[0].tolist()
    return enc.decode(output_tokens)