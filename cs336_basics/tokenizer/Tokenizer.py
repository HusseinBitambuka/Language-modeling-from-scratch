import time
import cProfile
import pstats
from BPE import BPE
from file_processing import find_chunk_boundaries

def train_tokenizer():
    file_path = "/home/husseinbitambuka/Dev/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    split_token = "<|endoftext|>".encode("utf-8")
    num_chunks = 8  # Tune as needed

    start = time.time()

    # Step 1: Initialize tokenizer
    bpe = BPE(vocab_size=vocab_size, special_tokens=special_tokens)

    # Step 2: Open file and find chunk boundaries
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks=num_chunks, split_special_token=split_token)

        # Step 3: For each chunk, decode and feed to BPE
        for start_offset, end_offset in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start_offset)
            chunk_bytes = f.read(end_offset - start_offset)
            chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
            bpe.add_chunk_to_pre_token_table(chunk_text)

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
