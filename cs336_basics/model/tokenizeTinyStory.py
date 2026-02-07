from cs336_basics.tokenizer import BPE2
import numpy as np
import time

tokenizer = BPE2.BPE.load(
    "/home/husseinbitambuka/Dev/assignment1-basics/cs336_basics/tokenizer/tokenizer_TinyStoriesV2-GPT4-train.pkl"
)







file_path = "/home/husseinbitambuka/Dev/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"

tokens = []

"""
with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        tokens.extend(tokenizer.encode(line))



"""
with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read(5_000_000)  # 5 MB

start = time.time()
tokens = tokenizer.encode(text)
print("Time:", time.time() - start)
print("Tokens:", len(tokens))

tokens_np = np.array(tokens, dtype=np.uint16)
np.save("TinyStoryTokens5MB.npy", tokens_np)
