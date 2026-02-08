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

"""

result:

(assignment1-basics) (base) husseinbitambuka@husseinbitambuka-HP-EliteDesk-705-G4-DM-35W-TAA:~/Dev/assignment1-basics/cs336_basics/model/perceptron$ uv run python -m cs336_basics.model.perceptron.MLPTrain
uint16 (1237571,)
epoch 0 | step 0 | loss 9.2095
epoch 0 | step 200 | loss 6.6797
epoch 0 | step 400 | loss 5.6475
epoch 0 | step 600 | loss 5.1675
epoch 0 | step 800 | loss 5.6474
epoch 0 | step 1000 | loss 5.5137
epoch 0 | step 1200 | loss 4.8919
epoch 0 | step 1400 | loss 4.6274
epoch 0 | step 1600 | loss 5.1553
epoch 0 | step 1800 | loss 4.3403
epoch 0 | step 2000 | loss 4.3945
epoch 0 | step 2200 | loss 3.7581
epoch 0 | step 2400 | loss 4.3386
epoch 0 | step 2600 | loss 4.8717
epoch 0 | step 2800 | loss 4.2395
epoch 0 | step 3000 | loss 3.9464
epoch 0 | step 3200 | loss 4.7259
epoch 0 | step 3400 | loss 4.4879
epoch 0 | step 3600 | loss 4.1604
epoch 0 | step 3800 | loss 3.8856
epoch 0 | step 4000 | loss 4.1215
epoch 0 | step 4200 | loss 3.4773
epoch 0 | step 4400 | loss 3.5894
epoch 0 | step 4600 | loss 3.5938
epoch 0 | step 4800 | loss 3.9340
epoch 0 | step 5000 | loss 4.0202
epoch 0 | step 5200 | loss 2.6743
epoch 0 | step 5400 | loss 3.8281
epoch 0 | step 5600 | loss 4.0419
epoch 0 | step 5800 | loss 4.2709
epoch 0 | step 6000 | loss 3.8778
epoch 0 | step 6200 | loss 3.7848
epoch 0 | step 6400 | loss 4.2536
epoch 0 | step 6600 | loss 3.7766
epoch 0 | step 6800 | loss 3.8606
epoch 0 | step 7000 | loss 3.5135
epoch 0 | step 7200 | loss 3.9785
epoch 0 | step 7400 | loss 3.2182
epoch 0 | step 7600 | loss 3.2800
epoch 0 | step 7800 | loss 4.0139
epoch 0 | step 8000 | loss 3.2160
epoch 0 | step 8200 | loss 3.7751
epoch 0 | step 8400 | loss 4.2048
epoch 0 | step 8600 | loss 3.6493
epoch 0 | step 8800 | loss 4.0882
epoch 0 | step 9000 | loss 3.2740
epoch 0 | step 9200 | loss 2.6732
epoch 0 | step 9400 | loss 3.6217
epoch 0 | step 9600 | loss 3.3007
epoch 0 | step 9800 | loss 3.4427
epoch 0 | step 10000 | loss 3.3364
epoch 0 | step 10200 | loss 3.7720
epoch 0 | step 10400 | loss 3.6662
epoch 0 | step 10600 | loss 2.9812
epoch 0 | step 10800 | loss 3.1007
epoch 0 | step 11000 | loss 2.9263
epoch 0 | step 11200 | loss 3.9106
epoch 0 | step 11400 | loss 3.2134
epoch 0 | step 11600 | loss 3.8967
epoch 0 | step 11800 | loss 3.3517
epoch 0 | step 12000 | loss 3.7880
epoch 0 | step 12200 | loss 3.6259
epoch 0 | step 12400 | loss 3.4128
epoch 0 | step 12600 | loss 3.3449
epoch 0 | step 12800 | loss 3.1181
epoch 0 | step 13000 | loss 3.5074
epoch 0 | step 13200 | loss 3.4609
epoch 0 | step 13400 | loss 3.7425
epoch 0 | step 13600 | loss 3.0805
epoch 0 | step 13800 | loss 3.1781
epoch 0 | step 14000 | loss 3.3652
epoch 0 | step 14200 | loss 3.6654
epoch 0 | step 14400 | loss 3.1425
epoch 0 | step 14600 | loss 3.3828
epoch 0 | step 14800 | loss 2.6851
epoch 0 | step 15000 | loss 3.2194
epoch 0 | step 15200 | loss 3.7045
epoch 0 | step 15400 | loss 3.6185
epoch 0 | step 15600 | loss 3.0497
epoch 0 | step 15800 | loss 3.3889
epoch 0 | step 16000 | loss 3.0803
epoch 0 | step 16200 | loss 3.7439
epoch 0 | step 16400 | loss 3.1333
epoch 0 | step 16600 | loss 3.6776
epoch 0 | step 16800 | loss 3.0433
epoch 0 | step 17000 | loss 3.1310
epoch 0 | step 17200 | loss 2.7369
epoch 0 | step 17400 | loss 2.5510
epoch 0 | step 17600 | loss 2.9899
epoch 0 | step 17800 | loss 3.4147
epoch 0 | step 18000 | loss 3.6866
epoch 0 | step 18200 | loss 3.1491
epoch 0 | step 18400 | loss 3.6249
epoch 0 | step 18600 | loss 3.7028
epoch 0 | step 18800 | loss 3.1800
epoch 0 | step 19000 | loss 3.2693
epoch 0 | step 19200 | loss 2.5453

Epoch 0 done | avg loss 3.6504 | time 1305.4s

epoch 1 | step 0 | loss 2.9156
epoch 1 | step 200 | loss 2.8158
epoch 1 | step 400 | loss 2.8632
epoch 1 | step 600 | loss 3.0138
epoch 1 | step 800 | loss 2.9336
epoch 1 | step 1000 | loss 2.9651
epoch 1 | step 1200 | loss 2.6722
epoch 1 | step 1400 | loss 2.8946
epoch 1 | step 1600 | loss 3.1320
epoch 1 | step 1800 | loss 2.8020
epoch 1 | step 2000 | loss 2.7808
epoch 1 | step 2200 | loss 2.6075
epoch 1 | step 2400 | loss 2.9344
epoch 1 | step 2600 | loss 3.3867
epoch 1 | step 2800 | loss 2.9128
epoch 1 | step 3000 | loss 2.7604
epoch 1 | step 3200 | loss 2.8460
epoch 1 | step 3400 | loss 2.9898
epoch 1 | step 3600 | loss 2.5774
epoch 1 | step 3800 | loss 2.9372
epoch 1 | step 4000 | loss 2.9772
epoch 1 | step 4200 | loss 2.6468
epoch 1 | step 4400 | loss 2.6778
epoch 1 | step 4600 | loss 2.7558
epoch 1 | step 4800 | loss 2.9319
epoch 1 | step 5000 | loss 3.1698
epoch 1 | step 5200 | loss 2.2653
epoch 1 | step 5400 | loss 2.6162
epoch 1 | step 5600 | loss 2.3216
epoch 1 | step 5800 | loss 2.5819
epoch 1 | step 6000 | loss 2.5069
epoch 1 | step 6200 | loss 3.4699
epoch 1 | step 6400 | loss 3.4442
epoch 1 | step 6600 | loss 2.8878
epoch 1 | step 6800 | loss 3.6691
epoch 1 | step 7000 | loss 2.8279
epoch 1 | step 7200 | loss 2.2282
epoch 1 | step 7400 | loss 2.6239
epoch 1 | step 7600 | loss 2.7852
epoch 1 | step 7800 | loss 2.6191
epoch 1 | step 8000 | loss 3.1087
epoch 1 | step 8200 | loss 2.6537
epoch 1 | step 8400 | loss 2.4994
epoch 1 | step 8600 | loss 2.6949
epoch 1 | step 8800 | loss 3.2644
epoch 1 | step 9000 | loss 2.7889
epoch 1 | step 9200 | loss 2.3878
epoch 1 | step 9400 | loss 3.2706
epoch 1 | step 9600 | loss 2.8360
epoch 1 | step 9800 | loss 2.9043
epoch 1 | step 10000 | loss 3.4032
epoch 1 | step 10200 | loss 2.8434
epoch 1 | step 10400 | loss 3.0910
epoch 1 | step 10600 | loss 3.0097
epoch 1 | step 10800 | loss 2.9918
epoch 1 | step 11000 | loss 3.6101
epoch 1 | step 11200 | loss 3.5802
epoch 1 | step 11400 | loss 3.0099
epoch 1 | step 11600 | loss 2.8202
epoch 1 | step 11800 | loss 2.6430
epoch 1 | step 12000 | loss 3.1931
epoch 1 | step 12200 | loss 3.1813
epoch 1 | step 12400 | loss 2.8563
epoch 1 | step 12600 | loss 3.0194
epoch 1 | step 12800 | loss 2.9428
epoch 1 | step 13000 | loss 3.3733
epoch 1 | step 13200 | loss 2.4795
epoch 1 | step 13400 | loss 3.0467
epoch 1 | step 13600 | loss 2.7697
epoch 1 | step 13800 | loss 3.5068
epoch 1 | step 14000 | loss 2.8539
epoch 1 | step 14200 | loss 2.9654
epoch 1 | step 14400 | loss 3.0275
epoch 1 | step 14600 | loss 3.0831
epoch 1 | step 14800 | loss 2.6480
epoch 1 | step 15000 | loss 2.8724
epoch 1 | step 15200 | loss 2.7238
epoch 1 | step 15400 | loss 3.2974
epoch 1 | step 15600 | loss 3.5116
epoch 1 | step 15800 | loss 2.2720
epoch 1 | step 16000 | loss 2.9622
epoch 1 | step 16200 | loss 2.1850
epoch 1 | step 16400 | loss 2.6331
epoch 1 | step 16600 | loss 3.3252
epoch 1 | step 16800 | loss 3.2673
epoch 1 | step 17000 | loss 3.4441
epoch 1 | step 17200 | loss 3.0274
epoch 1 | step 17400 | loss 2.8295
epoch 1 | step 17600 | loss 3.2219
epoch 1 | step 17800 | loss 2.6279
epoch 1 | step 18000 | loss 3.0355
epoch 1 | step 18200 | loss 2.9504
epoch 1 | step 18400 | loss 2.3821
epoch 1 | step 18600 | loss 3.4030
epoch 1 | step 18800 | loss 3.5433
epoch 1 | step 19000 | loss 2.8919
epoch 1 | step 19200 | loss 2.6027

Epoch 1 done | avg loss 2.9642 | time 1269.7s

epoch 2 | step 0 | loss 2.8315
epoch 2 | step 200 | loss 2.2206
epoch 2 | step 400 | loss 2.9440
epoch 2 | step 600 | loss 2.7983
epoch 2 | step 800 | loss 2.7991
epoch 2 | step 1000 | loss 2.5190
epoch 2 | step 1200 | loss 2.6581
epoch 2 | step 1400 | loss 3.0990
epoch 2 | step 1600 | loss 2.7475
epoch 2 | step 1800 | loss 2.5705
epoch 2 | step 2000 | loss 2.7586
epoch 2 | step 2200 | loss 2.7040
epoch 2 | step 2400 | loss 2.8622
epoch 2 | step 2600 | loss 2.8281
epoch 2 | step 2800 | loss 2.3393
epoch 2 | step 3000 | loss 2.3755
epoch 2 | step 3200 | loss 2.6384
epoch 2 | step 3400 | loss 2.8602
epoch 2 | step 3600 | loss 2.7956
epoch 2 | step 3800 | loss 3.1793
epoch 2 | step 4000 | loss 2.9285
epoch 2 | step 4200 | loss 2.3350
epoch 2 | step 4400 | loss 3.0392
epoch 2 | step 4600 | loss 2.8441
epoch 2 | step 4800 | loss 2.6614
epoch 2 | step 5000 | loss 2.5427
epoch 2 | step 5200 | loss 2.5863
epoch 2 | step 5400 | loss 2.5653
epoch 2 | step 5600 | loss 2.8598
epoch 2 | step 5800 | loss 2.8722
epoch 2 | step 6000 | loss 2.6389
epoch 2 | step 6200 | loss 2.8597
epoch 2 | step 6400 | loss 2.3836
epoch 2 | step 6600 | loss 3.1122
epoch 2 | step 6800 | loss 3.1444
epoch 2 | step 7000 | loss 2.3074
epoch 2 | step 7200 | loss 2.2679
epoch 2 | step 7400 | loss 2.9005
epoch 2 | step 7600 | loss 2.5150
epoch 2 | step 7800 | loss 3.1104
epoch 2 | step 8000 | loss 2.4852
epoch 2 | step 8200 | loss 3.0416
epoch 2 | step 8400 | loss 2.7047
epoch 2 | step 8600 | loss 2.4642
epoch 2 | step 8800 | loss 2.9494
epoch 2 | step 9000 | loss 2.6141
epoch 2 | step 9200 | loss 3.3381
epoch 2 | step 9400 | loss 2.4548
epoch 2 | step 9600 | loss 2.6710
epoch 2 | step 9800 | loss 2.3721
epoch 2 | step 10000 | loss 3.0811
epoch 2 | step 10200 | loss 2.7522
epoch 2 | step 10400 | loss 2.8604
epoch 2 | step 10600 | loss 2.4870
epoch 2 | step 10800 | loss 2.3342
epoch 2 | step 11000 | loss 2.4319
epoch 2 | step 11200 | loss 3.0546
epoch 2 | step 11400 | loss 3.0379
epoch 2 | step 11600 | loss 2.4832
epoch 2 | step 11800 | loss 2.7351
epoch 2 | step 12000 | loss 3.0053
epoch 2 | step 12200 | loss 2.3909
epoch 2 | step 12400 | loss 2.4999
epoch 2 | step 12600 | loss 3.3236
epoch 2 | step 12800 | loss 3.5236
epoch 2 | step 13000 | loss 2.4650
epoch 2 | step 13200 | loss 2.8606
epoch 2 | step 13400 | loss 2.3509
epoch 2 | step 13600 | loss 2.6311
epoch 2 | step 13800 | loss 3.1681
epoch 2 | step 14000 | loss 2.7875
epoch 2 | step 14200 | loss 3.0826
epoch 2 | step 14400 | loss 2.9651
epoch 2 | step 14600 | loss 2.4724
epoch 2 | step 14800 | loss 2.9451
epoch 2 | step 15000 | loss 2.6661
epoch 2 | step 15200 | loss 2.4395
epoch 2 | step 15400 | loss 2.7329
epoch 2 | step 15600 | loss 3.2537
epoch 2 | step 15800 | loss 2.6180
epoch 2 | step 16000 | loss 2.3536
epoch 2 | step 16200 | loss 3.3137
epoch 2 | step 16400 | loss 2.8818
epoch 2 | step 16600 | loss 2.5624
epoch 2 | step 16800 | loss 2.8622
epoch 2 | step 17000 | loss 2.5825
epoch 2 | step 17200 | loss 2.8333
epoch 2 | step 17400 | loss 2.7339
epoch 2 | step 17600 | loss 2.7899
epoch 2 | step 17800 | loss 2.7043
epoch 2 | step 18000 | loss 2.5710
epoch 2 | step 18200 | loss 2.6596
epoch 2 | step 18400 | loss 3.0743
epoch 2 | step 18600 | loss 2.7787
epoch 2 | step 18800 | loss 2.3428
epoch 2 | step 19000 | loss 2.5014
epoch 2 | step 19200 | loss 2.9952

Epoch 2 done | avg loss 2.7257 | time 1276.4s

epoch 3 | step 0 | loss 2.4980
epoch 3 | step 200 | loss 3.0017
epoch 3 | step 400 | loss 2.3213
epoch 3 | step 600 | loss 2.5330
epoch 3 | step 800 | loss 2.3763
epoch 3 | step 1000 | loss 2.4474
epoch 3 | step 1200 | loss 3.0625
epoch 3 | step 1400 | loss 2.6374
epoch 3 | step 1600 | loss 2.2785
epoch 3 | step 1800 | loss 2.3276
epoch 3 | step 2000 | loss 2.2622
epoch 3 | step 2200 | loss 2.5294
epoch 3 | step 2400 | loss 2.9226
epoch 3 | step 2600 | loss 2.2415
epoch 3 | step 2800 | loss 2.2454
epoch 3 | step 3000 | loss 2.9120
epoch 3 | step 3200 | loss 3.3081
epoch 3 | step 3400 | loss 2.8687
epoch 3 | step 3600 | loss 2.6803
epoch 3 | step 3800 | loss 2.5799
epoch 3 | step 4000 | loss 3.0476
epoch 3 | step 4200 | loss 2.4280
epoch 3 | step 4400 | loss 2.8661
epoch 3 | step 4600 | loss 2.3550
epoch 3 | step 4800 | loss 3.0128
epoch 3 | step 5000 | loss 2.4866
epoch 3 | step 5200 | loss 2.5547
epoch 3 | step 5400 | loss 2.9088
epoch 3 | step 5600 | loss 3.0079
epoch 3 | step 5800 | loss 2.9935
epoch 3 | step 6000 | loss 2.4666
epoch 3 | step 6200 | loss 2.4482
epoch 3 | step 6400 | loss 2.5514
epoch 3 | step 6600 | loss 2.5145
epoch 3 | step 6800 | loss 3.1611
epoch 3 | step 7000 | loss 2.5933
epoch 3 | step 7200 | loss 2.0556
epoch 3 | step 7400 | loss 2.8521
epoch 3 | step 7600 | loss 2.4513
epoch 3 | step 7800 | loss 2.1435
epoch 3 | step 8000 | loss 2.4489
epoch 3 | step 8200 | loss 1.9417
epoch 3 | step 8400 | loss 2.6677
epoch 3 | step 8600 | loss 2.1540
epoch 3 | step 8800 | loss 2.5217
epoch 3 | step 9000 | loss 2.6003
epoch 3 | step 9200 | loss 2.4126
epoch 3 | step 9400 | loss 2.2986
epoch 3 | step 9600 | loss 2.2560
epoch 3 | step 9800 | loss 2.9296
epoch 3 | step 10000 | loss 2.7678
epoch 3 | step 10200 | loss 2.5687
epoch 3 | step 10400 | loss 2.7343
epoch 3 | step 10600 | loss 2.6220
epoch 3 | step 10800 | loss 2.3362
epoch 3 | step 11000 | loss 2.6296
epoch 3 | step 11200 | loss 2.5751
epoch 3 | step 11400 | loss 2.1945
epoch 3 | step 11600 | loss 1.9383
epoch 3 | step 11800 | loss 2.8177
epoch 3 | step 12000 | loss 2.6686
epoch 3 | step 12200 | loss 2.8206
epoch 3 | step 12400 | loss 2.5801
epoch 3 | step 12600 | loss 3.0234
epoch 3 | step 12800 | loss 2.6811
epoch 3 | step 13000 | loss 2.5738
epoch 3 | step 13200 | loss 2.5807
epoch 3 | step 13400 | loss 2.1606
epoch 3 | step 13600 | loss 2.6276
epoch 3 | step 13800 | loss 2.1498
epoch 3 | step 14000 | loss 2.7165
epoch 3 | step 14200 | loss 3.1019
epoch 3 | step 14400 | loss 2.4407
epoch 3 | step 14600 | loss 2.4555
epoch 3 | step 14800 | loss 2.0733
epoch 3 | step 15000 | loss 2.3783
epoch 3 | step 15200 | loss 2.8634
epoch 3 | step 15400 | loss 2.3105
epoch 3 | step 15600 | loss 2.5548
epoch 3 | step 15800 | loss 2.3138
epoch 3 | step 16000 | loss 2.8107
epoch 3 | step 16200 | loss 2.2682
epoch 3 | step 16400 | loss 2.5185
epoch 3 | step 16600 | loss 2.7492
epoch 3 | step 16800 | loss 2.7108
epoch 3 | step 17000 | loss 3.5445
epoch 3 | step 17200 | loss 2.6244
epoch 3 | step 17400 | loss 2.8864
epoch 3 | step 17600 | loss 2.5378
epoch 3 | step 17800 | loss 2.5583
epoch 3 | step 18000 | loss 2.3521
epoch 3 | step 18200 | loss 2.4063
epoch 3 | step 18400 | loss 2.4156
epoch 3 | step 18600 | loss 2.7877
epoch 3 | step 18800 | loss 2.9755
epoch 3 | step 19000 | loss 2.6348
epoch 3 | step 19200 | loss 2.7705

Epoch 3 done | avg loss 2.5576 | time 1320.7s

epoch 4 | step 0 | loss 2.2619
epoch 4 | step 200 | loss 2.3095
epoch 4 | step 400 | loss 2.5881
epoch 4 | step 600 | loss 2.3832
epoch 4 | step 800 | loss 2.3732
epoch 4 | step 1000 | loss 2.5516
epoch 4 | step 1200 | loss 2.0790
epoch 4 | step 1400 | loss 2.3235
epoch 4 | step 1600 | loss 1.9867
epoch 4 | step 1800 | loss 2.4205
epoch 4 | step 2000 | loss 2.0044
epoch 4 | step 2200 | loss 2.1317
epoch 4 | step 2400 | loss 2.5431
epoch 4 | step 2600 | loss 1.7598
epoch 4 | step 2800 | loss 2.1573
epoch 4 | step 3000 | loss 2.4359
epoch 4 | step 3200 | loss 2.2327
epoch 4 | step 3400 | loss 2.7433
epoch 4 | step 3600 | loss 2.5843
epoch 4 | step 3800 | loss 2.9509
epoch 4 | step 4000 | loss 1.8839
epoch 4 | step 4200 | loss 2.3615
epoch 4 | step 4400 | loss 2.0586
epoch 4 | step 4600 | loss 2.5957
epoch 4 | step 4800 | loss 1.9532
epoch 4 | step 5000 | loss 2.4368
epoch 4 | step 5200 | loss 2.2861
epoch 4 | step 5400 | loss 1.9733
epoch 4 | step 5600 | loss 2.7954
epoch 4 | step 5800 | loss 2.4323
epoch 4 | step 6000 | loss 2.8299
epoch 4 | step 6200 | loss 2.5061
epoch 4 | step 6400 | loss 2.6884
epoch 4 | step 6600 | loss 2.3990
epoch 4 | step 6800 | loss 2.3275
epoch 4 | step 7000 | loss 2.5647
epoch 4 | step 7200 | loss 2.6359
epoch 4 | step 7400 | loss 2.4085
epoch 4 | step 7600 | loss 2.1877
epoch 4 | step 7800 | loss 2.3633
epoch 4 | step 8000 | loss 2.0453
epoch 4 | step 8200 | loss 2.7090
epoch 4 | step 8400 | loss 2.1776
epoch 4 | step 8600 | loss 2.6159
epoch 4 | step 8800 | loss 2.4071
epoch 4 | step 9000 | loss 2.3843
epoch 4 | step 9200 | loss 2.5346
epoch 4 | step 9400 | loss 1.8300
epoch 4 | step 9600 | loss 2.3150
epoch 4 | step 9800 | loss 1.8847
epoch 4 | step 10000 | loss 2.2170
epoch 4 | step 10200 | loss 2.2833
epoch 4 | step 10400 | loss 1.9094
epoch 4 | step 10600 | loss 2.7164
epoch 4 | step 10800 | loss 2.4021
epoch 4 | step 11000 | loss 2.7816
epoch 4 | step 11200 | loss 1.8787
epoch 4 | step 11400 | loss 2.5831
epoch 4 | step 11600 | loss 2.0765
epoch 4 | step 11800 | loss 2.6486
epoch 4 | step 12000 | loss 2.3027
epoch 4 | step 12200 | loss 2.6289
epoch 4 | step 12400 | loss 2.8780
epoch 4 | step 12600 | loss 2.5991
epoch 4 | step 12800 | loss 2.1479
epoch 4 | step 13000 | loss 2.5571
epoch 4 | step 13200 | loss 2.6358
epoch 4 | step 13400 | loss 2.6342
epoch 4 | step 13600 | loss 2.1837
epoch 4 | step 13800 | loss 2.4819
epoch 4 | step 14000 | loss 2.3298
epoch 4 | step 14200 | loss 3.0067
epoch 4 | step 14400 | loss 2.3737
epoch 4 | step 14600 | loss 2.2751
epoch 4 | step 14800 | loss 2.6285
epoch 4 | step 15000 | loss 2.4780
epoch 4 | step 15200 | loss 2.1665
epoch 4 | step 15400 | loss 2.9566
epoch 4 | step 15600 | loss 2.3043
epoch 4 | step 15800 | loss 2.8797
epoch 4 | step 16000 | loss 2.1391
epoch 4 | step 16200 | loss 1.8923
epoch 4 | step 16400 | loss 2.5456
epoch 4 | step 16600 | loss 3.8548
epoch 4 | step 16800 | loss 2.5427
epoch 4 | step 17000 | loss 2.4711
epoch 4 | step 17200 | loss 2.5749
epoch 4 | step 17400 | loss 2.1828
epoch 4 | step 17600 | loss 2.8268
epoch 4 | step 17800 | loss 2.7467
epoch 4 | step 18000 | loss 2.2453
epoch 4 | step 18200 | loss 2.4987
epoch 4 | step 18400 | loss 2.5100
epoch 4 | step 18600 | loss 2.5023
epoch 4 | step 18800 | loss 2.1565
epoch 4 | step 19000 | loss 2.4723
epoch 4 | step 19200 | loss 2.7248

Epoch 4 done | avg loss 2.4185 | time 1373.4s

tensor([  44,  956,  531,  259,  693,  458,  266, 5129,   46,  302,  693,  298,
        4552,  274, 3311,   46,  302,  693,  274,  266,  715,  584,  525,  522,
          46,  372,  584,  636,  525,  522,   46,   10,  500,  504,  501,   10,
         633,  650,  259,  562,   44,  570,  298,  259,  545,  734,  574,  583,
          46,  426,  480,  259,  440,   44, 1092,  730,   46,  426,  777,  273,
         397,  404,  421,  994,  274,  845])

=== GENERATED TEXT ===
, Spot saw a cat on the stage. The cat was dancing and singing. The cat and the dog were very happy. They were all very happy.
<|endoftext|>
Once upon a time, there was a little girl named Lily. She had a big, red ball. She loved to play with her toys and make
"""