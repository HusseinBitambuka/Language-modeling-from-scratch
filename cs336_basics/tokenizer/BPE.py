from collections import defaultdict
import regex as re
import heapq
import pickle

class BPE:
    def __init__(self, vocab_size: int, special_tokens: list[str]) -> None:
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merge_sets = {}
        self.special_token_to_id = {}
        
        for i, token in enumerate(special_tokens):
            token_id = 256 + i
            self.vocab[token_id] = token.encode("utf-8")
            self.special_token_to_id[token] = token_id

        self.next_token_id = 256 + len(special_tokens)
        self.pretoken_table_count = defaultdict(int)

    def add_chunk_to_pre_token_table(self, text_chunk: str) -> None:
        for pre_token in re.findall(self.PAT, text_chunk):
            if pre_token in self.special_token_to_id:
                continue
            token_bytes = tuple(pre_token.encode("utf-8"))
            self.pretoken_table_count[token_bytes] += 1

    def merge(self):
        pair_freq = defaultdict(int)
        pair_to_pretokens = defaultdict(set)

        for pretoken, freq in self.pretoken_table_count.items():
            for a, b in zip(pretoken, pretoken[1:]):
                pair = (a, b)
                pair_freq[pair] += freq
                pair_to_pretokens[pair].add(pretoken)
 
        heap = [(-count, pair) for pair, count in pair_freq.items()]
        heapq.heapify(heap)

        while len(self.vocab) < self.vocab_size and heap:
            _, most_frequent = heapq.heappop(heap)
            if most_frequent not in pair_freq:
                continue  # stale entry

            token1, token2 = most_frequent
            new_token_id = self.next_token_id
            self.next_token_id += 1

            new_token_bytes = self.vocab[token1] + self.vocab[token2]
            self.vocab[new_token_id] = new_token_bytes
            self.merge_sets[new_token_id] = (token1, token2)

            affected_pretokens = pair_to_pretokens.pop(most_frequent)
            new_table_updates = defaultdict(int)
            to_decrement = defaultdict(int)
            to_increment = defaultdict(int)

            for pretoken in affected_pretokens:
                freq = self.pretoken_table_count[pretoken]
                new_pre = []
                i = 0
                while i < len(pretoken):
                    if i < len(pretoken) - 1 and pretoken[i] == token1 and pretoken[i + 1] == token2:
                        new_pre.append(new_token_id)
                        i += 2
                    else:
                        new_pre.append(pretoken[i])
                        i += 1

                for a, b in zip(pretoken, pretoken[1:]):
                    to_decrement[(a, b)] += freq
                    pair_to_pretokens[(a, b)].discard(pretoken)

                new_pre_tuple = tuple(new_pre)
                new_table_updates[new_pre_tuple] += freq

                for a, b in zip(new_pre, new_pre[1:]):
                    to_increment[(a, b)] += freq
                    pair_to_pretokens[(a, b)].add(new_pre_tuple)

                del self.pretoken_table_count[pretoken]

            for pair, count in to_decrement.items():
                pair_freq[pair] -= count
                if pair_freq[pair] <= 0:
                    del pair_freq[pair]

            for pair, count in to_increment.items():
                pair_freq[pair] += count
                heapq.heappush(heap, (-pair_freq[pair], pair))

            self.pretoken_table_count.update(new_table_updates)

    def apply_merges(self, byte_sequence: list[int]) -> list[int]:
        merges = sorted(self.merge_sets.items())  # sort by token id (merge order)
        for token_id, (a, b) in merges:
            i = 0
            new_seq = []
            while i < len(byte_sequence):
                if i < len(byte_sequence) - 1 and byte_sequence[i] == a and byte_sequence[i + 1] == b:
                    new_seq.append(token_id)
                    i += 2
                else:
                    new_seq.append(byte_sequence[i])
                    i += 1
            byte_sequence = new_seq
        return byte_sequence

    def encode(self, text: str) -> list[int]:
        pre_tokens = re.findall(self.PAT, text)
        encoded_tokens = []
        for token in pre_tokens:
            if token in self.special_token_to_id:
                encoded_tokens.append(self.special_token_to_id[token])
            else:
                byte_sequence = list(token.encode("utf-8"))
                byte_sequence = self.apply_merges(byte_sequence)
                encoded_tokens.extend(byte_sequence)
        return encoded_tokens

    def decode(self, tokens: list[int]) -> str:
        byte_stream = b''.join(self.vocab.get(token, b'') for token in tokens)
        return byte_stream.decode("utf-8", errors="replace")

    def print_vocab(self):
        print("Vocabulary:")
        for tid in sorted(self.vocab.keys()):
            print(f"{tid}: {self.vocab[tid]}")

    def print_merges(self):
        print("Merge rules:")
        for tid, pair in sorted(self.merge_sets.items()):
            print(f"{tid}: {pair}")

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "vocab_size": self.vocab_size,
                "special_tokens": self.special_tokens,
                "vocab": self.vocab,
                "merge_sets": self.merge_sets,
                "special_token_to_id": self.special_token_to_id,
            }, f)

    @classmethod
    def load(cls, path: str) -> "BPE":
        with open(path, "rb") as f:
            state = pickle.load(f)
        bpe = cls(
            vocab_size=state["vocab_size"],
            special_tokens=state["special_tokens"]
        )
        bpe.vocab = state["vocab"]
        bpe.merge_sets = state["merge_sets"]
        bpe.special_token_to_id = state["special_token_to_id"]
        bpe.next_token_id = max(bpe.vocab) + 1
        return bpe
