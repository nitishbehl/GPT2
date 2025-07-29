import sys
import torch
import torch.nn.functional as F
import tiktoken

# Set device first
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Import EarlyExitClassifier after setting device
sys.path.append("/Users/nitishbehl/Desktop/gpt2_project/Phase3")
from early_exit import EarlyExitClassifier  # import EarlyExitClassifier

# Instantiate and load early exit classifier properly
early_exit_classifier = EarlyExitClassifier(hidden_size=384, vocab_size=50257)
state_dict = torch.load("/Users/nitishbehl/Desktop/gpt2_project/Phase3/checkpoints/early_exit_classifier.pt", map_location=device)
early_exit_classifier.load_state_dict(state_dict)
early_exit_classifier.to(device)
early_exit_classifier.eval()


# Add Phase1 directory to sys.path to import GPT model code
sys.path.append("/Users/nitishbehl/Desktop/gpt2_project/Phase1")
from train_gpt2 import GPT, GPTConfig

enc = tiktoken.get_encoding("gpt2")

# Paths to your saved checkpoints for each model
MODEL_PATHS = {
    "finetuned": "/Users/nitishbehl/Desktop/gpt2_project/Phase2/checkpoints/fine_tuned_gpt2_wikitext.pt",
    "pretrained": None,  # You can load pretrained weights if available
    "earlyexit": None,   # earlyexit uses separate classifier, no GPT checkpoint needed here
}

# Cache loaded models here
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
    model_instance = GPT(config)

    checkpoint_path = MODEL_PATHS.get(model_name)
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model_instance.load_state_dict(state_dict, strict=False)
    elif model_name != "earlyexit":
        # If no checkpoint and not earlyexit, warn and use random weights
        print(f"[Warning] No checkpoint for model '{model_name}', using random initialized weights.")

    model_instance.to(device)
    model_instance.eval()

    _loaded_models[model_name] = model_instance
    return model_instance

def generate_text(prompt="Hello, I am a language model,", model="finetuned", max_length=50):
    model_name = model.lower()
    print(f"[DEBUG] generate_text called with prompt='{prompt}' | model='{model_name}'")
    if model_name not in ["pretrained", "finetuned", "earlyexit"]:
        raise ValueError(f"Unknown model: {model}")

    model_instance = load_model(model_name) if model_name != "earlyexit" else load_model("finetuned")
    print(f"[DEBUG] Model loaded: {model_instance.__class__.__name__}") 

    tokens = enc.encode(prompt)
    print(f"[DEBUG] Tokenized input: {tokens}")
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        while x.size(1) < max_length:
            if model_name == "earlyexit":
                # For earlyexit, get all hidden states from GPT and pass layer9 output to classifier
                logits, all_hidden = model_instance(x, return_all_hidden=True)
                hidden9 = all_hidden[9]  # layer 9 output
                pred_logits = early_exit_classifier(hidden9)
                logits = pred_logits
                print("[DEBUG] Used early-exit classifier.") 
            else:
                logits, _ = model_instance(x)

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)

    decoded = enc.decode(x[0].tolist())
    print(f"[DEBUG] Generated output: {decoded}")
    return decoded
