# Encoder for Machine Translation 

In this section we will be implementing the encoder and testing the encoder part of a transformer model.

<img src="assets/encoder.png" width=70% ></img>

The encoder converts sequence of English tokens into contextual vectors that capture meaning and long range dependencies. 

The Encoder consists of a stack of identical layers, where each layer has two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feedforward network.

<img src="assets/encode-zoom.png" width=60%></img>

### Input data

- Tokenize english text (taking sections of word and give it a number)

- To process the test, the length of the training data needs to be the same. Padding (PAD) is pad_token_id (default 0).

- Then we use masks to tell the modelw hich tokens are real vs padding (1 for real tokens, 0 for PAD)

- Use learned positional embeddings 


###  Self attention: Multi-head self attention

Attention builds information onte token by creating query, key and Value for each token.

Q, K, V:
Q (Query) = what this word is looking for.
K (Key) = a tag that describes each word.
V (Value) = the information each word carries.



**Self attention** uses the information about the token (Q,K,V) to build relationships within the sentence. 

While **multi head attention** (scaled Dot product attention) builds multiple different types of relationships simutaneously. 

For example: 
- Head 1: Grammar relationships 
- Head 2: Object relationship
- Head 3: Contextual relationship
- etc...


<img src="assets/multi_head.png" width=50%></img>

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

Simple per token Multi-Layered Perceptron used to transform/refine the representations.


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


## Implementation 

**[encode.py](/encoder_transformer/encode.py)** file walks through these exact steps showing how to implement this in code. 


**Masked Language Modeling** is a great way to test encoding models by masking some of the tokens in a sentece and training to predict what the masked word should be. 
This is what I imlpemented in **[mlm.py](/encoder_transformer/mlm.py)** file.

To train and test the Masked Language Model: 
    
    #download the MLM data
    python3 setupdata.py 





