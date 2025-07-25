# app/main.py

from fastapi import FastAPI
import sys
import os

# Ensure root dir is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # This lets you import from the project root

from generate_text import generate_text  # This must import the generate_text function

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Use /generate?prompt=... to generate text."}

@app.get("/generate")
def generate(prompt: str):
    output = generate_text(prompt)
    return {"prompt": prompt, "generated": output}
