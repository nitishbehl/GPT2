# test_generate.py
import sys
sys.path.append("./Phase5")  # <-- Change this path if generate_text.py is in another folder

from generate_text import generate_text

prompt = "Hello how are you"
output = generate_text(prompt, model="finetuned")

print("Prompt:", prompt)
print("Output:", output)

