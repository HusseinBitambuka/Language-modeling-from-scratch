import time
import cProfile
import pstats
from tokenizer_parallelized import BPE



def train_tokenizer():
    file_path = "/home/husseinbitambuka/Dev/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    split_token = "<|endoftext|>".encode("utf-8")
    num_chunks =  3 # Tune as needed

    start = time.time()

    # Step 1: Initialize tokenizer
    bpe = BPE(vocab_size=vocab_size, special_tokens=special_tokens, num_process=3)
    bpe.add_pretokens()

    # Step 4: Train merges
    bpe.merge()

       # Step 5: Print stats
    elapsed = time.time() - start
    print(f"\nFinished training BPE tokenizer")
    print(f"Time elapsed: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    print(f"Final vocab size: {len(bpe.vocab)}")
    print(f"Merge operations: {len(bpe.merge_sets)}")

    # Step 6: Save tokenizer
    bpe.save("tokenizer_TinyStoriesV2-GPT4-train.pkl")
    print("Tokenizer saved to tokenizer_TinyStoriesV2-GPT4-train.pkl")

def profile_run():
    profile_path = "bpe_profile.prof"
    cProfile.run("train_tokenizer()", profile_path)
    print(f"\nProfile saved to {profile_path}")
    stats = pstats.Stats(profile_path)
    stats.sort_stats("cumulative").print_stats(20)

if __name__ == "__main__":
    profile_run()