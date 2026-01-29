# Byte-Level BPE Tokenizer: Implementation, Profiling, and Practical Constraints

This document describes the design, implementation, profiling results, and practical limitations encountered while training a byte-level Byte Pair Encoding (BPE) tokenizer on the TinyStories and OpenWebText datasets.

---

## 1. Objective

The goal of this project was to implement and train a **byte-level BPE tokenizer** from scratch, following the procedure outlined in the assignment. The tokenizer:

- Operates at the UTF-8 byte level
- Supports special tokens, including `<|endoftext|>`
- Learns BPE merges from byte-pair frequency statistics
- Serializes the learned vocabulary and merge rules to disk
- Supports encoding and decoding of text using the trained tokenizer

All experiments were conducted on a local development machine without GPUs.

---

## 2. Hardware Environment

All training and profiling experiments were performed on the following machine:

- **System Memory:** 15 GiB RAM
- **CPU:** AMD Ryzen 5 PRO 2400GE (4 cores / 8 threads)
- **Storage:** NVMe SSD

This hardware is sufficient for medium-scale experiments (e.g., TinyStories) but imposes strict memory limits for large-scale datasets such as OpenWebText.

---

## 3. Dataset Overview

### 3.1 TinyStories

- File size: ~2.2 GB
- Content: Short, simple, and repetitive natural language stories
- Document delimiter: `<|endoftext|>`

### 3.2 OpenWebText (OWT)

- File size: ~11.9 GB (training split)
- Content: Large-scale, heterogeneous web text including URLs, markup, code fragments, and long-tail vocabulary
- Document delimiter: `<|endoftext|>`

OpenWebText is more than **5× larger** than TinyStories and exhibits significantly higher lexical and structural diversity.

---

## 4. Tokenizer Design

### 4.1 Byte-Level Representation

The tokenizer operates directly on **UTF-8 bytes**, representing each pre-token as a sequence of integers in the range `[0, 255]`. This approach ensures:

- Full Unicode coverage
- No out-of-vocabulary issues
- Robust handling of rare or malformed text

All BPE merges are learned over byte sequences rather than Unicode characters or words.

---

### 4.2 Special Token Handling

The `<|endoftext|>` token is treated as a special case:

- It is detected during pretokenization
- It is explicitly added to the vocabulary
- No BPE merges are applied across this boundary

This ensures that document boundaries are respected and matches the assignment’s recommended approach.

---

### 4.3 Pretokenization

Pretokenization is performed using a regex-based splitter similar to those used in GPT-style tokenizers. The regex separates:

- Words
- Numbers
- Punctuation
- Whitespace

Pretokenization is applied independently to each document segment delimited by `<|endoftext|>`.

---

## 5. Training Procedure

Training proceeds in two main phases:

### 5.1 Pretoken Frequency Collection

The dataset is first split into chunks aligned on `<|endoftext|>` boundaries. Pretokenization and byte conversion are then applied to each chunk, producing a frequency table of byte sequences.

To reduce runtime, **multiprocessing** is used during this phase:

- Each worker processes a disjoint chunk of the data
- Local frequency tables are merged into a global table
- The number of worker processes is intentionally limited (2–3) to avoid excessive memory usage

This phase dominates both runtime and memory consumption.

---

### 5.2 BPE Merge Training

After collecting pretoken frequencies:

- Adjacent byte-pair frequencies are computed
- The most frequent pair is merged
- Frequency tables are updated incrementally
- The process repeats until the target vocabulary size is reached

Compared to pretokenization, the merge loop itself is relatively fast.

---

## 6. Profiling Results

Profiling was performed using `cProfile`.

### Key Result

> The majority of training time is spent in the **pretokenization phase**, specifically in regex-based splitting, UTF-8 decoding, and construction of byte-sequence frequency tables.

The BPE merge loop accounts for only a small fraction of total runtime.

This result is consistent with known BPE implementations and explains why production tokenizers often implement pretokenization in native code.

---

## 7. Experimental Results

### 7.1 TinyStories

- Vocabulary size: 10,000
- Training time: ~7 minutes
- Memory usage: Within system limits
- Longest tokens: Multi-byte or multi-word sequences reflecting frequent patterns in simple narrative text

The longest tokens learned from TinyStories are semantically coherent and expected given the dataset’s repetitive structure.

---

### 7.2 OpenWebText: Practical Constraints

Training on the **full OpenWebText training set (~11.9 GB)** was not feasible on the available hardware.

The pretokenization phase requires holding large frequency tables in memory, and effective memory usage scales far beyond the raw file size due to:

- UTF-8 decoding
- Regex tokenization
- Python object overhead
- Intermediate frequency tables

On a system with 15 GB of RAM, the operating system terminated the process due to memory pressure before training could complete. This behavior is expected and aligns with the assignment’s stated upper bound of **≤100 GB RAM** for OpenWebText training.

---

### 7.3 OpenWebText Observations (Subset Training)

Training on a large subset of OpenWebText reveals clear qualitative differences compared to TinyStories:

- Significantly longer tokens
- Tokens corresponding to URLs, markup, and structured web artifacts
- Greater vocabulary diversity and noise

These observations reflect the heterogeneous nature of web-scale text and support the expected comparison between the two datasets.

---

## 8. Encoding and Decoding

The trained tokenizer supports both encoding and decoding:

- **Encoding**:
  1. Pretokenize input text
  2. Convert pre-tokens to UTF-8 byte sequences
  3. Apply BPE merges in the order learned during training

- **Decoding**:
  - Concatenate byte sequences corresponding to token IDs
  - Decode the resulting byte stream as UTF-8

This mirrors the training procedure and ensures correctness.

---

## 9. Engineering Reflections

This project highlights several important systems-level insights:

- Pretokenization is the dominant cost in BPE training
- Python multiprocessing improves throughput but increases memory pressure
- Excessive parallelism can lead to OS-level termination due to memory exhaustion
- Large-scale BPE training typically requires native implementations with shared memory and multithreading

While such optimizations are common in production tokenizers, the Python implementation is sufficient for the scope of this assignment.

---

## 10. Conclusion

The implemented byte-level BPE tokenizer successfully trains on the TinyStories dataset and demonstrates expected qualitative behavior when applied to OpenWebText. Profiling confirms that pretokenization dominates runtime, and practical hardware constraints limit full-scale training on very large datasets. Overall, the implementation satisfies the assignment requirements and provides insight into the trade-offs involved in tokenizer training at scale.
