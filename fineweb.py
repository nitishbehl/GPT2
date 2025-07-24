
import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm

# ---------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)  # 100M tokens per shard
MAX_SHARDS = 10

# Create local directory if it doesn't exist
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Download the dataset (streaming)
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)

# Init GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

# Tokenization function
def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    return np.array(tokens, dtype=np.uint16)

def main():
    global all_tokens_np, token_count, shard_index, progress_bar
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    shard_index = 0
    nprocs = 4 
    progress_bar = None

    with mp.Pool(nprocs) as pool:
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)

                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))

            else:
                filename = os.path.join(DATA_CACHE_DIR, f"eduFineweb_train_{shard_index:03d}.bin")
                remainder = shard_size - token_count
                all_tokens_np[token_count:shard_size] = tokens[:remainder]
                with open(filename, "wb") as f:
                    f.write(all_tokens_np.tobytes())
                print(f" Saved {shard_size} tokens to {filename}")
                shard_index += 1
                
                # Enforce MAX_SHARDS limit
                if shard_index >= MAX_SHARDS:
                    print(f"Reached MAX_SHARDS = {MAX_SHARDS}. Stopping.")
                    break
                progress_bar = None
                all_tokens_np[:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

            

    # Save last shard
    if token_count > 0:
        filename = os.path.join(DATA_CACHE_DIR, f"eduFineweb_train_{shard_index:03d}.bin")
        with open(filename, "wb") as f:
            f.write(all_tokens_np[:token_count].tobytes())
        print(f" Saved {token_count} tokens to {filename} (final shard)")

if __name__ == '__main__':
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass  # Context already set — ignore error
    main()

