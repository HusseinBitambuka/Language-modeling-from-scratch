# MLP Language Model — Training and Results

## Overview

This document summarizes the training process and results of a Multi-Layer Perceptron (MLP) language model trained on a 5MB subset of the TinyStories dataset.

The goal of this experiment was to:

- Train a next-token prediction model
- Validate the training pipeline end-to-end
- Establish a baseline before transitioning to more advanced architectures

---

## Dataset

- Source: TinyStories (subset)
- Training size: ~5MB of text
- Tokenizer: Byte Pair Encoding (BPE)
- Vocabulary size: 10,000
- Total tokens: 1,237,571
- Token dtype: `uint16` (converted to `int64` for training)

### Training Sample Construction

A sliding window dataset was used.

For a context length of 16:

[t0, t1, ..., t15] → t16
[t1, t2, ..., t16] → t17
...

Number of training samples:

~1,237,571 - 16

---

## Model Configuration

- Context length: 16
- Embedding dimension (`d_model`): 128
- Hidden dimension: 512
- Batch size: 64
- Optimizer: Adam
- Learning rate: 3e-4
- Loss: CrossEntropyLoss
- Device: CPU (based on training time logs)

---

## Training Performance

### Epoch-Level Results

| Epoch | Average Loss | Time (seconds) |
| ----- | ------------ | -------------- |
| 0     | 3.6504       | 1305.4         |
| 1     | 2.9642       | 1269.7         |
| 2     | 2.7257       | 1276.4         |
| 3     | 2.5576       | 1320.7         |
| 4     | 2.4185       | 1373.4         |

### Observations

- Initial loss started around **9.2**
- Rapid decrease during first epoch
- Stabilized near **2.4** by epoch 4
- Loss steadily decreased across epochs
- No instability or divergence observed

---

## Training Time

Average time per epoch:

~21–23 minutes

Total training time (5 epochs):

~1 hour 45 minutes

All training was performed on CPU.

---

## Generation Results

After training, the model was used in greedy decoding mode:

- Prompt: random context from dataset
- Generation length: 50 tokens
- Decoding: BPE decode

### Sample Output

, Spot saw a cat on the stage. The cat was dancing and singing. The cat and the dog were very happy. They were all very happy.
<|endoftext|>
Once upon a time, there was a little girl named Lily. She had a big, red ball. She loved to play with her toys and make

### Observations

- Model learned basic sentence structure
- Maintained grammatical coherence
- Repeated common TinyStories phrasing patterns
- Demonstrated clear next-token prediction capability
- Limited long-range coherence (expected for MLP)

---

## Interpretation of Results

### What the Model Learned

- Local syntactic patterns
- Frequent story templates
- Common phrase continuation patterns
- Basic grammar and punctuation

### Limitations Observed

- No deep narrative coherence
- No long-range dependency modeling
- Some repetition
- Context window limited to 16 tokens

These limitations are expected due to:

- Fixed context window
- No recurrence
- No attention mechanism

---

## Baseline Established

This MLP provides:

- A verified training pipeline
- A functioning autoregressive generation loop
- A stable optimization setup
- A quantitative baseline (loss ≈ 2.4)

This baseline will be used for comparison when introducing:

- Normalization
- Residual connections
- Positional embeddings
- Attention mechanisms
- Transformer blocks

---

## Conclusion

The MLP successfully:

- Trained without instability
- Reduced loss consistently
- Generated coherent short text
- Demonstrated effective embedding + feedforward modeling

This completes the baseline architecture stage and prepares the transition toward Transformer-based models.
