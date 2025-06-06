import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16      
block_size = 64      
max_iters = 5000      
eval_interval = 50   
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 40
n_embd = 128         
n_head = 4           
n_layer = 4         
dropout = 0.15       
# ------------

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'input.txt')

torch.manual_seed(1337)
# Reading the data and loading it in
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Character to integers mapping
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: string -> integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: integers -> string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # data is moved to the device (GPU)
    return x, y

@torch.no_grad() # Do not track gradients here - allows for memory efficiency
def estimate_loss(): 
    out = {}
    model.eval() # Model is placed into evaluation mode with no training-specific behaviour such as dropout, etc
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item() # Saves the loss for it's respective iteration
        out[split] = losses.mean() # Save average losses to either train or val
    model.train()
    return out

class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # lower triangular matrix

        self.dropout = nn.Dropout(dropout) # dropout layer for regularisation
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        v = self.value(x) # (B,T,C)

        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T) and using scaled attention
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # zero out the upper triangular part
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei) # apply dropout to the attention weights

        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate the outputs of all heads
        out = self.proj(out)
        return out  # (B,T,C) output of the multi-head attention layer

class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # first linear layer
            nn.ReLU(), # activation function
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x) # apply the feedforward network

class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: number of attention heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # self-attention layer
        self.ffwd = FeedForward(n_embd) # feedforward network
        self.ln1 = nn.LayerNorm(n_embd) # layer normalisation after self-attention
        self.ln2 = nn.LayerNorm(n_embd) # layer normalisation after feedforward network

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # uses residual connections (layer normalisation is applied before self-attention)
        x = x + self.ffwd(self.ln2(x)) # uses residual connections (layer normalisation is applied before feedforward network)
        return x

# Bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # block size will get it's own embedding vector
        self.block = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) 
        self.ln_f = nn.LayerNorm(n_embd) # final layer normalisation
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C) elementwise addition of token and position embeddings
        x = self.block(x) # (B,T,C) pass through the transformer blocks
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # (B, T) get the last block_size tokens
            # get the predictions and loss
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device) # model is moved to the device (GPU)

# create a PyTorch optimiser
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0: # Checks for remainder 
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # context is created on the device (GPU)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
