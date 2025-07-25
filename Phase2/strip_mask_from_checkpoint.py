import torch

checkpoint_path = "checkpoints/fine_tuned_gpt2_wikitext.pt"
new_path = "checkpoints/fine_tuned_gpt2_wikitext_stripped.pt"

# Load the full checkpoint
state_dict = torch.load(checkpoint_path, map_location="cpu")

# Remove all keys that end with "attn.mask"
cleaned_state_dict = {k: v for k, v in state_dict.items() if not k.endswith("attn.mask")}

# Save the cleaned model
torch.save(cleaned_state_dict, new_path)
print(" Saved cleaned checkpoint to", new_path)
