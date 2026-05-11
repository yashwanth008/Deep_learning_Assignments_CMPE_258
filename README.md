# Sequence, Language, Vision & Graph Models — Colab Portfolio

> **Course Assignment** | Executed Jupyter Notebooks with Video Walkthroughs
> All notebooks are fully executed with outputs. Each has a companion video walkthrough.

---

## Table of Contents

| # | Notebook | Topics | Video |
|---|----------|--------|-------|
| 1 | [RNN, LSTM, GRU & WaveNet](#1-sequence-models-rnn--lstm--gru--wavenet) | Vanilla RNN, LSTM gates, GRU, Dilated Convolutions | [▶ Watch](YOUR_VIDEO_LINK_1) |
| 2 | [10 Years of NLP](#2-10-years-of-deep-learning-in-nlp) | Word2Vec, Seq2Seq, Attention, Transformers, BERT, GPT, RLHF | [▶ Watch](YOUR_VIDEO_LINK_2) |
| 3 | [Vision Transformers](#3-vision-transformers--the-frontier-of-computer-vision) | ViT, CLIP, DINOv2, SAM, Hybrid Architectures | [▶ Watch](YOUR_VIDEO_LINK_3) |
| 4 | [GNN Fundamentals](#4-graph-neural-networks-fundamentals) | Graphs, Message Passing, GCN from Scratch, Node Classification | [▶ Watch](YOUR_VIDEO_LINK_4) |

---

## Repository Structure

```
cmpe258-sequence-language-vision-graph/
│
├── README.md                                              ← You are here
│
└── Colabs/
    ├── final_rnn_lstm_gru_wavenet_zero_to_hero.ipynb
    ├── final_nlp_deep_learning_10_years_tutorial.ipynb
    ├── final_vision_transformers_tutorial.ipynb
    └── final_gnn_fundamentals_tutorial.ipynb
```

---

## 1. Sequence Models: RNN · LSTM · GRU · WaveNet

### What This Notebook Covers

A ground-up journey through recurrent and convolutional sequence architectures, all implemented in PyTorch. Inspired by Aurélien Géron's *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*.

| Part | Topics |
|------|--------|
| **Part I** | The World of Sequences — why order matters, variable-length inputs |
| **Part II** | Vanilla RNN — the simplest recurrent network, hidden state, BPTT |
| **Part III** | LSTM — Long Short-Term Memory, cell state, forget/input/output gates |
| **Part IV** | GRU — Gated Recurrent Unit, reset & update gates, LSTM vs GRU |
| **Part V** | Deep RNNs & Practical Tricks — stacking, dropout, bidirectional, gradient clipping |
| **Part VI** | WaveNet — causal & dilated convolutions, exponential receptive field |
| **Part VII** | Grand Comparison — benchmarks, loss curves, when to use what |

### Key Concepts Demonstrated
- The vanishing gradient problem and why plain RNNs forget over long sequences
- How LSTM gates solve vanishing gradients with an additive cell state update
- GRU as a parameter-efficient alternative to LSTM
- How dilated causal convolutions let WaveNet train in parallel with a large receptive field
- All four architectures benchmarked on the same character-level language modeling task

### Libraries Used
`torch` · `numpy` · `matplotlib`

---

## 2. 10 Years of Deep Learning in NLP

### What This Notebook Covers

A visual, beginner-friendly tour of 50 key concepts across a decade of NLP progress — from tokenization and word embeddings all the way to ChatGPT and RLHF.

```
Chapter 1: The Basics       Chapter 2: Sequential Models     Chapter 3: Transformers
├── Tokenization            ├── RNNs                         ├── Self-Attention
└── Word Embeddings         ├── LSTMs & GRUs                 ├── Multi-Head Attention
                            ├── Seq2Seq / Encoder-Decoder    ├── Positional Encoding
                            └── Bahdanau Attention           └── Full Architecture

Chapter 4: Large Language Models              Chapter 5: Human Alignment
├── GPT (Decoder-only)                        ├── Hallucination Problem
├── BERT (Encoder-only)                       ├── InstructGPT & RLHF
├── XLNet & T5                                └── ChatGPT & GPT-4
└── LoRA & Knowledge Distillation
```

### Key Concepts Demonstrated
- Three levels of tokenization (character, word, subword/BPE) built from scratch
- Word2Vec embeddings and the king − man + woman ≈ queen analogy
- Seq2Seq architecture and the fixed-size context vector bottleneck
- How Bahdanau attention lets the decoder query all encoder states at each step
- Transformer self-attention, multi-head attention, and positional encodings
- BERT (bidirectional, masked LM) vs GPT (autoregressive, next-token prediction)
- LoRA: fine-tuning large models with < 1% of their parameters
- RLHF pipeline: supervised fine-tuning → reward model → PPO optimization

### Libraries Used
`numpy` · `matplotlib` · `seaborn` · `torch` · `transformers`

---

## 3. Vision Transformers & The Frontier of Computer Vision

### What This Notebook Covers

How the Transformer — designed for text — was adapted for images, and the wave of powerful models it enabled.

| Chapter | Topics |
|---------|--------|
| **Chapter 1** | Attention Mechanism — scaled dot-product, multi-head attention |
| **Chapter 2** | Vision Transformer (ViT) — patch embeddings, CLS token, positional encoding |
| **Chapter 3** | CLIP — contrastive vision-language pretraining, zero-shot classification |
| **Chapter 4** | DINOv2 — self-supervised ViT, teacher-student training, emergent segmentation |
| **Chapter 5** | SAM — promptable segmentation (point / box / text → mask) |
| **Chapter 6** | Hybrid Architectures — ConvNeXt, Swin Transformer |
| **Chapter 7** | Practical Applications — using SOTA models for real tasks |

### Key Concepts Demonstrated
- Self-attention formula: `Attention(Q,K,V) = softmax(QK^T / √d_k) · V`
- Patch tokenization: a 224×224 image becomes 196 tokens of size 16×16
- Why ViT needs large datasets (no inductive bias like CNNs)
- CLIP's contrastive loss on NxN image-text similarity matrices
- DINOv2's emergent object segmentation without any segmentation labels
- SAM's three-part architecture: image encoder, prompt encoder, mask decoder

### Libraries Used
`torch` · `torchvision` · `numpy` · `matplotlib` · `PIL`

---

## 4. Graph Neural Networks: Fundamentals

### What This Notebook Covers

Graph Neural Networks from zero — no prior GNN experience needed. Part 1 of a 3-part series.

| Chapter | Topics |
|---------|--------|
| **Chapter 1** | What Are Graphs? — nodes, edges, real-world examples across domains |
| **Chapter 2** | Graph Representations — adjacency matrix, edge list, adjacency list |
| **Chapter 3** | Why GNNs? — failure modes of standard ML on graphs |
| **Chapter 4** | Message Passing Paradigm — Message → Aggregate → Update |
| **Chapter 5** | GCN from Scratch — NumPy implementation of `H' = σ(D⁻¹/²AD⁻¹/²HW)` |
| **Chapter 6** | GCN in PyTorch — clean, trainable implementation |
| **Chapter 7** | Hands-On Demo — node classification on the Karate Club dataset |

### Key Concepts Demonstrated
- Graphs as a universal data structure: social networks, molecules, citation graphs, knowledge graphs
- Why adjacency matrices are memory-inefficient for large sparse graphs
- The message passing framework that underlies all modern GNNs
- GCN's symmetric normalized adjacency: prevents high-degree nodes from dominating
- How 2 GCN layers give every node information from its 2-hop neighborhood
- Node classification demo: learned embeddings cleanly separate two communities

> **Note:** This is Part 1 of 3. Parts 2 and 3 cover GraphSAGE, GAT, GIN, molecular property prediction, and knowledge graph completion.

### Libraries Used
`numpy` · `matplotlib` · `networkx` · `torch` · `scikit-learn`

---

## Learning Summary

By completing all four notebooks, the following areas of modern deep learning are covered end-to-end:

```
Sequence · Language · Vision · Graph
│
├── Sequence Models
│   └── RNN / LSTM / GRU / WaveNet   → How neural nets process ordered data
│
├── Language Models
│   └── 10 Years of NLP              → Word2Vec → Transformers → ChatGPT
│
├── Vision Models
│   └── Vision Transformers          → ViT · CLIP · DINOv2 · SAM
│
└── Graph Models
    └── GNN Fundamentals             → Message passing · GCN · Node classification
```

---

## Tech Stack

**Framework:** PyTorch 2.x · **Environment:** Google Colab (GPU)  
**Libraries:** NumPy · Matplotlib · Seaborn · NetworkX · HuggingFace Transformers

---

*Last updated: May 2026*
