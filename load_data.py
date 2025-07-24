from datasets import load_dataset

print("Running load_data.py...")
print("Loading dataset from Hugging Face...")

dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
print("Printing first 3 samples:")
for i, example in enumerate(dataset):
    print(example)
    if i >= 2:
        break

