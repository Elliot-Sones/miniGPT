# 🚀 Transformers: From Scratch to Machine Translation

A comprehensive implementation of Transformer architectures, built from the ground up to understand every component. This project demonstrates the complete journey from basic attention mechanisms to full machine translation systems.

![Transformer Architecture](assets/transformer.png)

## 📚 Table of Contents

1. [What is a Transformer?](#what-is-a-transformer)
2. [Encoder vs Decoder Components](#encoder-vs-decoder-components)
3. [Project Structure & Implementation](#project-structure--implementation)
4. [Quick Start Guide](#quick-start-guide)
5. [Technical Deep Dive](#technical-deep-dive)

---

## 🤖 What is a Transformer?

A **Transformer** is a deep neural network architecture that revolutionized natural language processing by using **attention mechanisms** to process sequences of data (like text) in parallel rather than sequentially.

### Key Concepts

**🔍 Attention Mechanism**: The core innovation that allows the model to focus on different parts of the input sequence when processing each element.

**🧠 Multi-Head Attention**: Multiple attention mechanisms running in parallel, each learning different types of relationships between words.

**📐 Self-Attention**: A mechanism where each position in a sequence can attend to all positions in the same sequence to compute a representation.

### Mathematical Foundation

The attention mechanism is computed as:

```
Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

Where:
- **Q** (Query): What the current word is looking for
- **K** (Key): Tags that describe each word  
- **V** (Value): The actual information each word carries
- **d_k**: Dimension of the key vectors (for scaling)

### Why Transformers Matter

- **Parallel Processing**: Unlike RNNs, all positions are processed simultaneously
- **Long-Range Dependencies**: Can capture relationships between distant words
- **Scalability**: Efficient training on large datasets
- **Transfer Learning**: Pre-trained models can be fine-tuned for specific tasks

---

## ⚖️ Encoder vs Decoder Components

### 🔍 Encoder Component

**Purpose**: Processes the input sequence and creates rich contextual representations.

**Key Features**:
- **Self-Attention**: Each word attends to all other words in the input
- **Bidirectional Processing**: Can see both past and future context
- **Contextual Encoding**: Converts input tokens into meaningful vector representations

**Architecture**:
```
Input → Embedding → Multi-Head Self-Attention → Feed Forward → Output
```

**What it learns**:
- Grammatical relationships (subject-verb, adjective-noun)
- Semantic meaning and context
- Long-range dependencies between words

### 🔮 Decoder Component

**Purpose**: Generates output sequences based on encoder representations and previous outputs.

**Key Features**:
- **Masked Self-Attention**: Can only attend to previous positions (causal masking)
- **Cross-Attention**: Attends to encoder outputs for translation context
- **Autoregressive Generation**: Generates one token at a time

**Architecture**:
```
Input → Embedding → Masked Self-Attention → Cross-Attention → Feed Forward → Output
```

**What it learns**:
- Language generation patterns
- Translation mappings from source to target
- Sequential dependencies in output language

### 🔄 How They Work Together

1. **Encoder** processes the source sentence (e.g., English)
2. **Decoder** uses encoder's context to generate target sentence (e.g., French)
3. **Cross-attention** allows decoder to focus on relevant parts of source
4. **Teacher Forcing** during training helps decoder learn correct patterns

---

## 🏗️ Project Structure & Implementation

This project is organized into three main components, each building upon the previous:

### 📁 Project Overview

```
Transformers/
├── encoder_transformer/     # Encoder implementation & MLM training
├── decoder_transformer/     # Decoder implementation & language modeling  
├── machine_translation/     # Full seq2seq system combining both
├── assets/                  # Visual diagrams and checkpoints
└── app.py                   # Production web interface
```

---

### 🔍 Encoder Transformer (`encoder_transformer/`)

**Purpose**: Implements and trains the encoder component using Masked Language Modeling (MLM).

#### Key Files:

**`encode.py`** - Core Encoder Architecture
- **`EncoderConfig`**: Configuration dataclass with hyperparameters
- **`TokenPositionalEmbedding`**: Combines token and positional embeddings
- **`MultiHeadSelfAttention`**: Implements scaled dot-product attention with padding masks
- **`FeedForward`**: Position-wise MLP for each token
- **`EncoderBlock`**: Complete encoder layer with residual connections and layer norm
- **`Encoder`**: Full encoder stack with multiple layers

**Key Features**:
- Pre-LayerNorm architecture for training stability
- Proper padding mask handling for variable-length sequences
- Weight initialization for optimal training
- Support for different vocabulary sizes and model dimensions

**`mlm/train.py`** - MLM Training Pipeline
- **`ProductionTokenizer`**: Custom tokenizer for the target sentence
- **`MLMDataset`**: Dataset that repeats target sentence for training
- **`create_mlm_targets`**: Creates masked language modeling targets
- **`ProductionMLM`**: Complete MLM model using the encoder
- **Training loop**: 20 epochs with 20% masking probability

**What it demonstrates**:
- How encoders learn contextual representations
- Masked language modeling as a training objective
- Word relationship learning through attention

#### Usage:
```bash
# Train the MLM model
python encoder_transformer/mlm/train.py

# Test the trained model
python encoder_transformer/mlm/test.py
```

---

### 🔮 Decoder Transformer (`decoder_transformer/`)

**Purpose**: Implements the decoder component for autoregressive language generation.

#### Key Files:

**`training.py`** - GPT-style Language Model Training
- **`Head`**: Single attention head implementation
- **`MultiHeadAttention`**: Multiple attention heads in parallel
- **`FeedForward`**: Position-wise feedforward network
- **`Block`**: Complete transformer block with residual connections
- **`GPTLanguageModel`**: Full decoder-only language model
- **Training infrastructure**: Checkpointing, EMA, learning rate scheduling

**Key Features**:
- Causal masking for autoregressive generation
- Weight tying between input embeddings and output layer
- PyTorch SDPA integration for optimized attention
- Comprehensive training loop with evaluation and checkpointing

**What it demonstrates**:
- How decoders generate text autoregressively
- Causal attention mechanisms
- Language modeling as a training objective

#### Usage:
```bash
# Train the decoder model
python decoder_transformer/training.py

# Generate text samples
python decoder_transformer/sample.py --prompt "ROMEO:" --max_new_tokens 300
```

---

### 🌐 Machine Translation (`machine_translation/`)

**Purpose**: Combines encoder and decoder into a complete sequence-to-sequence translation system.

#### Key Files:

**`mini_transformer.py`** - Complete Seq2Seq Architecture
- **`Seq2SeqConfig`**: Configuration for the full model
- **`MultiHeadAttention`**: Flexible attention for both self and cross-attention
- **`Encoder`**: Source language encoder
- **`Decoder`**: Target language decoder with cross-attention
- **`Seq2Seq`**: Complete translation model
- **`greedy_generate`**: Inference with greedy decoding
- **`prepare_decoder_inputs_and_labels`**: Helper for teacher forcing

**Key Features**:
- Unified attention mechanism for both encoder and decoder
- Cross-attention between decoder and encoder outputs
- Teacher forcing during training
- Greedy and sampling-based generation
- Proper handling of special tokens (BOS, EOS, PAD)

**`setup_data.py`** - Data Preparation Pipeline
- Downloads WMT14/16 or OPUS Books datasets
- Text normalization and cleaning
- Length and ratio filtering
- Deterministic train/test splitting
- CSV export for training

**`train_mini.py`** - Translation Training
- **`CsvPairs`**: Dataset class for parallel text pairs
- **`collate_batch`**: Batch collation with proper padding
- **Training loop**: With evaluation, checkpointing, and early stopping
- **`preview_samples`**: Sample translation generation during training

**What it demonstrates**:
- How encoder and decoder work together
- Cross-attention mechanisms
- Sequence-to-sequence learning
- Machine translation pipeline

#### Usage:
```bash
# Prepare translation data
python machine_translation/setup_data.py --dataset wmt14 --out_dir data/en_fr

# Train the translation model
python machine_translation/train_mini.py --train_csv data/en_fr/train.csv --val_csv data/en_fr/test.csv

# Run translation inference
python machine_translation/translate.py --checkpoint checkpoints/best.pt --input "Hello world"
```

---

### 🌐 Production Interface (`app.py`)

**Purpose**: Web interface for demonstrating the trained MLM model.

**Features**:
- **Gradio-based UI**: Interactive web interface
- **Word Prediction**: Mask any word in the target sentence
- **Confidence Scores**: Shows prediction confidence and top alternatives
- **Model Loading**: Automatically loads trained production model

**What it demonstrates**:
- Production deployment of transformer models
- Interactive MLM word prediction
- Real-time inference capabilities

---

## 🚀 Quick Start Guide

### Prerequisites

```bash
pip install torch torchvision torchaudio
pip install datasets pandas tqdm
pip install tokenizers
pip install gradio
```

### 1. Train the Encoder (MLM)

```bash
cd encoder_transformer/mlm
python train.py
```

This trains a masked language model on the sentence: *"This model create relationships between the words to learn what word is missing!"*

### 2. Train the Decoder (Language Model)

```bash
cd decoder_transformer
python training.py
```

This trains a GPT-style language model on text data.

### 3. Train Machine Translation

```bash
# First, prepare the data
python machine_translation/setup_data.py --dataset wmt14 --out_dir data/en_fr

# Then train the translation model
python machine_translation/train_mini.py --train_csv data/en_fr/train.csv --val_csv data/en_fr/test.csv
```

### 4. Launch Production Interface

```bash
python app.py
```

Visit the provided URL to interact with the trained MLM model.

---

## 🔬 Technical Deep Dive

### Attention Mechanism Implementation

The core attention mechanism is implemented in `MultiHeadAttention`:

```python
def forward(self, x, kv=None, key_padding_mask=None, causal=False):
    # Project to Q, K, V
    q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(kv).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(kv).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
    
    # Compute attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
    
    # Apply masks
    if key_padding_mask is not None:
        attn_scores = attn_scores.masked_fill(~keep_mask, float("-inf"))
    if causal:
        causal_mask = torch.ones((T, S), device=device).triu(1)
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
    
    # Softmax and weighted sum
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v)
    return self.out_proj(attn_output)
```

### Key Design Decisions

1. **Pre-LayerNorm**: Applied before attention and feedforward for training stability
2. **Residual Connections**: Help with gradient flow and training
3. **Weight Tying**: Reduces parameters by sharing input/output embeddings
4. **Proper Masking**: Handles padding and causal constraints correctly
5. **Scaled Attention**: Division by √d_k prevents attention weights from becoming too peaked

### Training Strategies

- **Teacher Forcing**: During training, decoder sees correct previous tokens
- **Label Smoothing**: Reduces overconfidence and improves generalization
- **Learning Rate Scheduling**: Warmup followed by decay for stable training
- **Gradient Clipping**: Prevents exploding gradients
- **Checkpointing**: Saves model state for resuming training

---

## 🎯 What You'll Learn

By working through this project, you'll understand:

1. **Attention Mechanisms**: How self-attention and cross-attention work
2. **Transformer Architecture**: The complete encoder-decoder structure
3. **Training Objectives**: MLM, language modeling, and sequence-to-sequence learning
4. **Implementation Details**: Proper masking, residual connections, and normalization
5. **Production Deployment**: How to create interactive interfaces for trained models

This project provides a complete foundation for understanding and implementing transformer architectures from scratch! 🚀
