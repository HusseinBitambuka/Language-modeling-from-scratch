import time
import cProfile
import pstats
from BPE2 import BPE
from file_processing import find_chunk_boundaries
from collections import defaultdict
import regex as re
import multiprocessing



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

class BPEPara:
    def __init__(
            self, vocab_size:int,
            special_tokens:list[str],
            file_path:str="/home/husseinbitambuka/Dev/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
            num_process:int = multiprocessing.cpu_count()
            
            ) -> None:
        
        self.vocab_size:int = vocab_size
        self.num_process = num_process
        self.special_tokens:list[str] = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.file_path:str = file_path
        self.vocab:dict[int, bytes] = {i:bytes([i])for i in range(256)}
        self.merge_sets:dict[int, tuple[int, int]] = {}
        self.special_token_to_id:dict[str, int] = {}

        for i, token in enumerate(special_tokens):
            token_id = 256 + i
            self.vocab[token_id] = token.encode("utf-8")
            self.special_token_to_id[token] = token_id

        self.next_token_id = 256 + len(special_tokens)
        self.pretoken_table_count:dict[tuple[int, ...], int] = defaultdict(int)
        self.reader = open(self.file_path, "rb")
        
    def process_chunk(self, start_offset:int, end_offset:int):
        local_counts:dict[tuple[int,...], int] = defaultdict(int)
        self.reader.seek(start_offset)
        chunck_bytes:bytes = self.reader.read(end_offset - start_offset)
        text:str = chunck_bytes.decode("utf-8", errors="ignore")
        bytes_cached:dict[str, tuple[int, ...]] = {}

        for pre_token in re.findall(self.PAT, text):
            if pre_token in self.special_token_to_id:
                continue
            cached = bytes_cached.get(pre_token)
            if cached is None:
                cached = tuple(pre_token.encode("utf-8"))
                bytes_cached[pre_token] = cached
            local_counts[cached] += 1
        return local_counts
