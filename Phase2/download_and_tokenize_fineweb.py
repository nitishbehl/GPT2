import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Constants
local_dir = "edu_fineweb10B"
remote_name = "CC-MAIN-2013-20"
shard_size = int(1e8)  # 100M tokens

# Globals for tokenizer (used in child processes)
enc = None
eot = None

def init_tokenizer():
    global enc, eot
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    global enc, eot
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    return np.array(tokens, dtype=np.uint16)

def main():
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)

    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    shard_index = 0
    progress_bar = None
    nprocs = 4

    with mp.Pool(nprocs, initializer=init_tokenizer) as pool:
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
                progress_bar = None

                if shard_index >= 20:
                    print("Stopping after 20 shards.")
                    break
                all_tokens_np[:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

    if token_count > 0:
        filename = os.path.join(DATA_CACHE_DIR, f"eduFineweb_train_{shard_index:03d}.bin")
        with open(filename, "wb") as f:
            f.write(all_tokens_np[:token_count].tobytes())
        print(f" Saved {token_count} tokens to {filename} (final shard)")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
