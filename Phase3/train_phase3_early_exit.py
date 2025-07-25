import torch
import sys
sys.path.append("../Phase2")
from train_phase2_wikitext import GPT, GPTConfig, get_batch, x_data
from early_exit import EarlyExitClassifier

# Load GPT2 fine-tuned model from Phase 2
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = GPT(GPTConfig()).to(device)
model.load_state_dict(torch.load("checkpoints/fine_tuned_gpt2_wikitext.pt", map_location=device))
model.eval()

# Init Early Exit Classifier
classifier = EarlyExitClassifier().to(device)
optimizer = torch.optim.AdamW(classifier.parameters(), lr=3e-4)

steps = 100
grad_accum = 2

print("Training early-exit classifier...\n")
for step in range(steps):
    optimizer.zero_grad()
    total_loss = 0.0

    for _ in range(grad_accum):
        x, y = get_batch(step)
        if x is None: break
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            _, _, block9_out = model(x, targets=y, return_block9=True)

        logits = classifier(block9_out)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = loss / grad_accum
        loss.backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
    optimizer.step()
    print(f"Step {step:03d} | Loss: {total_loss:.4f}")

# Save classifier
torch.save(classifier.state_dict(), "checkpoints/early_exit_classifier.pt")
print("\n Early-exit classifier saved to checkpoints/early_exit_classifier.pt")
