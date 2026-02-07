import numpy as np
from tokenizers import Tokenizer

TOKENIZER_PATH = "hf_tinystories_bpe.json"
DATA_PATH = "/home/husseinbitambuka/Dev/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"

# Load tokenizer
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# Stream file (DO NOT read whole file into memory)
token_ids = []

with open(DATA_PATH, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        encoding = tokenizer.encode(line)
        token_ids.extend(encoding.ids)

# Convert to NumPy
tokens_np = np.array(token_ids, dtype=np.int16)

# Save
np.save("TinyStories_tokens_float32.npy", tokens_np)

print("Total tokens:", tokens_np.shape[0])
print("Saved to TinyStories_tokens_float32.npy")
