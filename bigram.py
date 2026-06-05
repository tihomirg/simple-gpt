import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters #
# ---------------------------------------------------------------------------------------------------------------- #
batch_size = 32 # number of sequences that will be processed in parallel
block_size = 8 # max context length
train_set_size = 0.9 # x 100%
max_iters = 3000 # number of training iterations
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
n_embd = 32
# ---------------------------------------------------------------------------------------------------------------- #

# device
#device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"

print(f"Device: {device}")

def device_synchorize():
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

torch.manual_seed(1337)

# get the file
# win: wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# mac: curl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt > input.txt

# let's read the "input.txt" file
with open("input.txt", mode='r', encoding="utf-8") as f:
    text = f.read()

#let's find all the unique characters in our dataset
chars = sorted(set(text))
vocab_size = len(chars)

# let's create simple tokenizer (encoder), but also detokenizer (decoder)

ch2i = {char:i for i, char in enumerate(chars)}
i2ch = {i:char for i, char in enumerate(chars)}

# encoder takes string as input and returns list of numbers (tokens)
encode = lambda s : [ch2i[ch] for ch in s]

# decoder takes list of numbers (tokens) and return string
decode = lambda tokens : "".join([i2ch[i] for i in tokens])

# load data into a tensor
data = torch.tensor(encode(text), dtype=torch.long)

# split data into train and val sets
n = int(train_set_size * len(data))
train_data = data[:n]
val_data = data[n:]

# batch of random slices of size (batch_size, block_size)
# note that the tragets y are like the inputs x, just with offset of plus one (next token in sequence)
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(high=len(data) - block_size, size=(batch_size, )) # randomly choses batch_size indexes from the range [0, len(data) - block_size))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # every token directly reads the logits of the next token from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targtets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # token embeddings (B, T, C) where C = n_embd
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C) where C = n_embd
        x = tok_emb + pos_emb # due to brodcasting of pos_emb to (B, T, C) we get that x has also dimensions (B, T, C)
        logits = self.lm_head(x) # logits (B, T, C) where C = vocab_size
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C) # (B*T, C)
            targets = targets.view(B * T) # (B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_next_tokens):

        # idx - is the context that has (B, T) dimensions

        for _ in range(max_next_tokens):
            
            # make a prediction
            logits, _ = self(idx) # (B, T, C)

            # focus onto the last element
            logits = logits[:, -1, :] # (B, C)

            probs = F.softmax(logits, dim=-1) # (B, C)

            index = torch.multinomial(probs, num_samples=1) # (B, 1)

            idx = torch.cat((idx, index), dim=1) # (B, T + 1)
        
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

# create torch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# function that estimates losses
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# training
for iter in range(max_iters):

    # after first 10 iterations we start to measure the time such that we can compare 
    if iter == 10:
        device_synchorize()
        start_time = time.perf_counter()

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample the batch of data
    xb, yb = get_batch("train")

    # do forward pass and update loss 
    _, loss = m(xb, yb)
    
    # set grads to None
    optimizer.zero_grad(set_to_none=True)
    
    # do backward pass and calculate grads
    loss.backward()

    # update parameters
    optimizer.step()


# benchmaring 
device_synchorize()
end_time = time.perf_counter()

total_time = end_time - start_time
print(f"Training time on {device} is {total_time}")

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_next_tokens=500)[0].tolist()))