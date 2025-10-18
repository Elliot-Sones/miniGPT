# Encoder for Machine Translation 

In this section we will be implementing the encoder.

<img src="assets/encoder.png" width=70% ></img>

The encoder converts sequence of English tokens into contextual vectors that capture meaning and long range dependencies. 

The Encoder consists of a stack of identical layers, where each layer has two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feedforward network.

<img src="assets/encode-zoom.png" width=60%></img>

### Input data

- Tokenize english text to src_input_ids(B,T) 

Padding (PAD) is pad_token_id (default 0).

- Build src_attention_mask (B, T) with 1 for real tokens, 0 for PAD.

- Use learned positional embeddings 


Q, K, V:
Q (Query) = what this word is looking for.
K (Key) = a tag that describes each word.
V (Value) = the information each word carries.


###  Self attention: Multi-head attention

<img src="assets/multi_head.png" width=60%></img>

$$
Multihead(V,K,Q)= Concat(head_1,...,head_h)W^O

$$
$$
    where: head_i= attention(QW^Q_i, KW^Q_i )
$$

- Make Q (what I’m looking for), K (my tag), V (my info)
- Compare Q to every K → relevance scores
- Scale scores, mask PAD keys, then softmax → focus weights
- Mix: weighted average of all V → the head output

### Layer 2: Feed foward

Simple per token MLP used to transform/reine the representation


### Residuals + layernorm

We the add a residual connection around each sub layers and followed by layer normalization. 

Residual: 
After ever sublayer in the encoder (MHA + FFN layers), we add attention output back into x.
$$
x = x + SelfAttention(LN(x))
$$
$$
x = x + FeedForward(LN(x))
$$

$x = LN(x)$ at the end of the encoder stack





