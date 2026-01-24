import argparse
import os
import numpy as np
import mmap
from BPE import BPE

def tokenize_large_file_to_numpy(
    tokenizer,
    file_path: str,
    save_path: str,
    split_token: str = "<|endoftext|>",
    chunk_size: int = 64 * 1024 * 1024,  # 64MB
):
    token_ids = []
    split_id = tokenizer.special_token_to_id.get(split_token)

    with open(file_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        offset = 0
        file_size = mm.size()

        while offset < file_size:
            end = min(offset + chunk_size, file_size)
            mm.seek(offset)
            chunk_bytes = mm.read(end - offset)
            text = chunk_bytes.decode("utf-8", errors="ignore")
            ids = tokenizer.encode(text)
            token_ids.extend(ids)
            if split_id is not None:
                token_ids.append(split_id)
            offset = end

        mm.close()

    arr = np.array(token_ids, dtype=np.uint16)
    np.save(save_path, arr)
    print(f"Saved token array ({len(arr)} tokens) to {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Tokenize a large file with a trained BPE tokenizer.")
    parser.add_argument("file_path", help="Path to the text file to tokenize.")
    parser.add_argument(
        "--tokenizer", default="tokenizer_TinyStoriesV2-GPT4-train.pkl",
        help="Path to the trained BPE tokenizer pickle file."
    )
    args = parser.parse_args()

    # Load tokenizer
    bpe = BPE.load(args.tokenizer)

    # Get save path (same directory as tokenizer)
    tokenizer_dir = os.path.dirname(os.path.abspath(args.tokenizer))
    file_stem = os.path.splitext(os.path.basename(args.file_path))[0]
    output_path = os.path.join(tokenizer_dir, f"{file_stem}_tokens.npy")

    tokenize_large_file_to_numpy(
        tokenizer=bpe,
        file_path=args.file_path,
        save_path=output_path
    )


if __name__ == "__main__":
    main()
