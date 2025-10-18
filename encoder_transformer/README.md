# Encoder for Machine Translation 

In this section we will be implementing the encoder.

<img src="assets/encoder.png" width=70% ></img>

The encoder converts sequence of English tokens into contextual vectors that capture meaning and long range dependencies

We will do this by implementing 

<img src="assets/encode-zoom.png" width=60%></img>

Encoder (English):
Stack of N encoder blocks: self-attention + FFN, residuals, layer norms.
Inputs: src_input_ids, src_attention_mask (to mask PAD).
Output: encoder_hidden_states (context for cross-attention).


## Steps and connections

- Implement `encoder.py` with embeddings, N encoder blocks, final LayerNorm.
- Inputs: `src_input_ids`, `src_attention_mask` → outputs `encoder_hidden_states`.
- Connect as keys/values for decoder cross-attention in `machine_translation`.
- Ensure `embed_dim % num_heads == 0` for head splits.
- Unit-test shapes and PAD masking with tiny tensors before integration.
