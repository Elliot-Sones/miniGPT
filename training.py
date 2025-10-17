import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
from contextlib import nullcontext

# ========== Hyperparameters
batch_size = 64
block_size = 256 # We will predict the 257 token on the basis of the 256 before that now!
max_iters = 5000
eval_interval = 1000
learning_rate = 3e-4 # Bring down the learning rate
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 50
n_embd = 384 # 384 / 6 = 64
n_head = 6
n_layer = 6
dropout = 0.1

# LR schedule (cosine with warmup)
warmup_iters = 200
lr_decay_iters = max_iters
min_lr = 1e-4
# ==========================
torch.manual_seed(1337)

# Dataset
with open('archive/train.csv', 'r', encoding='UTF-8') as f:
    text = f.read()

# Tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
lookup_table_in = { ch:i for i,ch in enumerate(chars)}
lookup_table_out = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [lookup_table_in[c] for c in s] # Encoder
decode = lambda l: ''.join([lookup_table_out[i] for i in l]) # Decoder
data = torch.tensor(encode(text), dtype=torch.long)

# Train and Test Split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Data Loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# =========== Transformer Components:

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout) # <----- Added Dropout!

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei) # <------ Added Dropout!
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # Added the possibility to add heads per parameter and loop. That's it.
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout) # <----- More Dropout!

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # <----- More Dropout!
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout), # <----- More Dropout!
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# We now don't have a BigramLanguage anymore
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Added the possibility to add heads per parameter and loop. That's it.
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # Weight tying: improves perplexity and reduces params slightly
        self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # Compute CE loss in float32 for stability with mixed precision
            loss = F.cross_entropy(logits.float(), targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
# Train =============================
model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
print(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1, betas=(0.9, 0.95))

# autocast context for faster MPS execution
ctx = torch.autocast(device_type='mps', dtype=torch.float16) if device == 'mps' else nullcontext()

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / max(1, warmup_iters)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

log_window = 100
t_last = time.time()

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets (skip step 0)
    if iter > 0 and (iter % eval_interval == 0 or iter == max_iters - 1):
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # update learning rate via schedule
    lr = get_lr(iter)
    for g in optimizer.param_groups:
        g['lr'] = lr

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    with ctx:
        logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # progress logging: avg ms/iter over last window and ETA
    if (iter + 1) % log_window == 0:
        t_now = time.time()
        ms_per_iter = (t_now - t_last) * 1000.0 / log_window
        t_last = t_now
        remaining = max_iters - (iter + 1)
        eta_min = (remaining * ms_per_iter) / 1000.0 / 60.0
        print(f"~{ms_per_iter:.1f} ms/iter, ETA {eta_min:.1f} min, lr {lr:.2e}")
