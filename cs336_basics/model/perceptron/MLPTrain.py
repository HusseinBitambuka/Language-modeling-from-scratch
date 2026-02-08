from MLP1 import MLPLanguageModel
from ...tokenizer.BPE2 import BPE
from torch.utils.data import DataLoader
import numpy as np
import torch
import time

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, context_len):
        self.tokens = tokens
        self.context_len = context_len

    def __len__(self):
        return len(self.tokens) - self.context_len

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.context_len]
        y = self.tokens[idx + self.context_len]
        return x, y


tokens_np = np.load("/home/husseinbitambuka/Dev/assignment1-basics/cs336_basics/model/TinyStoryTokens5MB.npy") 
print(tokens_np.dtype, tokens_np.shape)

# Convert once to torch (keep on CPU for now)
tokens = torch.from_numpy(tokens_np.astype(np.int64))

context_len = 16
batch_size = 64

dataset = TokenDataset(tokens, context_len)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLPLanguageModel(
    vocab_size=10_000,
    context_len=context_len,
    d_model=128,
    hidden_dim=512,
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()

def train(model, loader, epochs=3):
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        start = time.time()

        for step, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)                # (batch, vocab)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 200 == 0:
                print(
                    f"epoch {epoch} | step {step} | loss {loss.item():.4f}"
                )

        elapsed = time.time() - start
        avg_loss = total_loss / len(loader)
        print(
            f"\nEpoch {epoch} done | avg loss {avg_loss:.4f} | time {elapsed:.1f}s\n"
        )

train(model, loader, epochs=5)

@torch.no_grad()
def generate(model, start_tokens, max_new_tokens=50):
    model.eval()

    tokens = start_tokens.clone().to(device)

    for _ in range(max_new_tokens):
        x = tokens[-context_len:].unsqueeze(0)
        logits = model(x)
        next_token = torch.argmax(logits, dim=-1)
        tokens = torch.cat([tokens, next_token], dim=0)

    return tokens
# pick a random starting point
i = torch.randint(0, len(tokens) - context_len, (1,)).item()
prompt = tokens[i : i + context_len]

generated = generate(model, prompt)

print(generated)

# ---- BPE decoding ----

tokenizer = BPE.load(
    "/home/husseinbitambuka/Dev/assignment1-basics/cs336_basics/tokenizer/tokenizer_TinyStoriesV2-GPT4-train.pkl"
)

generated_ids = generated.tolist()
decoded = tokenizer.decode(generated_ids)
print("\n=== GENERATED TEXT ===")
print(decoded)
