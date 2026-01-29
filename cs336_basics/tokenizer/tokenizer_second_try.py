import time
import cProfile
import pstats
from BPE2 import BPE
from file_processing import find_chunk_boundaries

def train_tokenizer():
    file_path = "/home/husseinbitambuka/Dev/assignment1-basics/data/owt_train.txt"
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
    bpe.save("owt_train.txt.pkl")
    print("Tokenizer saved to tokenizer_owt_train.txt.pkl")

def profile_run():
    profile_path = "bpe_profile.prof"
    cProfile.run("train_tokenizer()", profile_path)
    print(f"\nProfile saved to {profile_path}")
    stats = pstats.Stats(profile_path)
    stats.sort_stats("cumulative").print_stats(20)

if __name__ == "__main__":
    profile_run()
'''
Finished training BPE tokenizer
Time elapsed: 395.90 seconds (6.60 minutes)
Final vocab size: 10000
Merge operations: 9743
Tokenizer saved to tokenizer_TinyStoriesV2-GPT4-train.pkl

Profile saved to bpe_profile.prof
Sun Jan 25 22:34:09 2026    bpe_profile.prof

         7846284 function calls (7846187 primitive calls) in 396.387 seconds

   Ordered by: cumulative time
   List reduced from 179 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000  396.380  396.380 {built-in method builtins.exec}
        1    0.463    0.463  396.380  396.380 <string>:1(<module>)
        1    0.934    0.934  395.917  395.917 /home/husseinbitambuka/Dev/assignment1-basics/cs336_basics/tokenizer/tokenizer_second_try.py:7(train_tokenizer)
        8  157.962   19.745  378.760   47.345 /home/husseinbitambuka/Dev/assignment1-basics/cs336_basics/tokenizer/BPE2.py:25(add_chunk_to_pre_token_table)
        8    0.000    0.000  220.734   27.592 /home/husseinbitambuka/Dev/assignment1-basics/.venv/lib/python3.12/site-packages/regex/regex.py:331(findall)
        8  220.731   27.591  220.731   27.591 {method 'findall' of '_regex.Pattern' objects}
        1    3.968    3.968    5.930    5.930 /home/husseinbitambuka/Dev/assignment1-basics/cs336_basics/tokenizer/BPE2.py:33(merge)
        8    5.350    0.669    5.350    0.669 {method 'decode' of 'bytes' objects}
       15    4.928    0.329    4.928    0.329 {method 'read' of '_io.BufferedReader' objects}
  1463260    0.421    0.000    0.421    0.000 {method 'add' of 'set' objects}
   132434    0.420    0.000    0.420    0.000 {built-in method _heapq.heappop}
  1369818    0.400    0.000    0.400    0.000 {method 'discard' of 'set' objects}
  3146163    0.342    0.000    0.342    0.000 {built-in method builtins.len}
  1367451    0.221    0.000    0.221    0.000 {method 'append' of 'list' objects}
   265430    0.076    0.000    0.076    0.000 {built-in method _heapq.heappush}
     9743    0.070    0.000    0.070    0.000 {method 'update' of 'dict' objects}
    59923    0.064    0.000    0.064    0.000 {method 'encode' of 'str' objects}
     9743    0.008    0.000    0.008    0.000 {method 'pop' of 'dict' objects}
        1    0.007    0.007    0.007    0.007 {method 'disable' of '_lsprof.Profiler' objects}
        5    0.007    0.001    0.007    0.001 {built-in method builtins.print}
'''