from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# Paths
DATA_PATH = "/home/husseinbitambuka/Dev/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
TOKENIZER_PATH = "hf_tinystories_bpe.json"

# Create tokenizer
tokenizer = Tokenizer(BPE(unk_token="<unk>"))

# Byte-level pretokenization (GPT-style)
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

# Trainer
trainer = BpeTrainer(
    vocab_size=10_000,
    min_frequency=2,
    special_tokens=[
        "<|endoftext|>",
        "<unk>",
    ],
)

# Train (this is FAST)
tokenizer.train([DATA_PATH], trainer)

# Save tokenizer
tokenizer.save(TOKENIZER_PATH)

print("Tokenizer trained and saved to:", TOKENIZER_PATH)
