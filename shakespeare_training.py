# GPT-3 Paper 
# add cosing delay  
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from datetime import datetime

# Device setup
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "training_logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"shakespeare_training_{timestamp}.log")),
        logging.StreamHandler()
    ]
)

# Initial logging messages
logging.info("=" * 70)
logging.info("SHAKESPEARE GPT-2 TRAINING")
logging.info("=" * 70)

# Log system info
logging.info(f"Device: {device}")
logging.info(f"PyTorch version: {torch.__version__}")
logging.info("=" * 50)

def save_model(model, optimizer, loss, output_dir="saved_models"):
    """Save the model, optimizer state, and training info"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gpt_model_{timestamp}.pt"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare state dict
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model.config,
        'loss': loss
    }
    
    # Save the model
    torch.save(save_dict, filepath)
    print(f"Model saved to {filepath}")
    return filepath

def save_checkpoint(model, optimizer, scheduler, loss, step, timestamp):
    """Save model checkpoint with timestamp"""
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(save_dir, f"shakespeare_model_{timestamp}_step{step}.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal = True) # Flash attention

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50304 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)



    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# model = GPT.from_pretrained('gpt2')

# SEED
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# STOP
num_return_sequences = 5
max_length = 30



import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2') 
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')
        logging.info(f"Loaded {len(self.tokens)} tokens.")
        logging.info(f"1 epoch = {len(self.tokens) // (B * T)} batches.")
        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B*T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# CHANGES IN CURRENT CODE
torch.set_float32_matmul_precision('high')
model = GPT(GPTConfig())
model.to(device)
# model = torch.compile(model)

# CODE UPDATE HERE
max_lr = 6e-4 
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 25000

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <=1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

train_loader = DataLoaderLite(B = 8, T = 1024)

# NEW CODE
import time
# optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)
for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # NEW CODE ADDED HERE
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y) 
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
    # NEW CODE
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    optimizer.step()
    torch.cuda.synchronize() 
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f'step{step} | loss: {loss.item()} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec: .2f} | norm: {norm:.2f}')
    logging.info(f'step{step} | loss: {loss.item()} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec: .2f} | norm: {norm:.2f}')
    if loss.item() < 0.099999:
        print("Target Loss Reached < 0.099999")
        logging.info("Target Loss Reached < 0.099999. Stopping training.")
        break
saved_path = save_model(model, optimizer, loss)
print(f"Model saved to: {saved_path}")
logging.info(f"Model saved to: {saved_path}")
logging.info(f"Final Loss: {loss.item()}")

print(loss)
import sys; sys.exit(0)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x)[0] # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

# Model initialization and training setup
model = GPT.from_pretrained('gpt2-medium')  # 350M parameters
model.to(device)

# Log model details
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Initialized GPT-2 Medium with {num_params:,} parameters")

# Training hyperparameters
batch_size = 16
seq_length = 128
grad_accum_steps = 4
effective_batch = batch_size * grad_accum_steps
logging.info(f"Training config: batch_size={batch_size}, seq_length={seq_length}, grad_accum_steps={grad_accum_steps}")
logging.info(f"Effective batch size: {effective_batch}")

train_loader = DataLoaderLite(B=batch_size, T=seq_length)

# Optimizer and scheduler setup
initial_lr = 2e-5
min_lr = 1e-6
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=initial_lr,
    weight_decay=0.1,
    betas=(0.9, 0.95)
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=1000,  # warmup steps
    T_mult=2,
    eta_min=min_lr
)

# Training state
best_loss = float('inf')
target_loss = 0.099999
patience = 0
max_patience = 15
max_steps = 50000
grad_clip = 1.0

logging.info(f"Starting training with target loss: {target_loss}")
logging.info("=" * 50)

# Training loop
optimizer.zero_grad()

for step in range(max_steps):
    model.train()
    total_loss = 0
    
    # Gradient accumulation
    for accum_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        with torch.cuda.amp.autocast(enabled=True):
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
        
        loss.backward()
        total_loss += loss.item() * grad_accum_steps
    
    # Gradient clipping and optimization step
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    # Logging (every 10 steps)
    if step % 10 == 0:
        lr = optimizer.param_groups[0]['lr']
        logging.info(f"Step: {step:6d} | Loss: {total_loss:.6f} | LR: {lr:.2e}")
    
    # Save best model
    if total_loss < best_loss:
        best_loss = total_loss
        patience = 0
        save_checkpoint(model, optimizer, scheduler, total_loss, step, timestamp)
        logging.info(f"New best loss achieved: {best_loss:.6f}")
    else:
        patience += 1
    
    # Check target loss
    if total_loss < target_loss:
        logging.info(f"🎉 Target loss achieved! Final loss: {total_loss:.6f}")
        save_checkpoint(model, optimizer, scheduler, total_loss, step, timestamp)
        break
    
    # Early stopping
    if patience >= max_patience:
        logging.info(f"Training stopped after {max_patience} steps without improvement")
        break

logging.info("=" * 50)
logging.info(f"Training completed. Best loss: {best_loss:.6f}")
logging.info(f"Final checkpoint saved at step {step}")

# Log detailed model configuration
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info("Model Configuration:")
logging.info(f"- Model: GPT-2 Medium")
logging.info(f"- Total Parameters: {num_params:,}")
logging.info(f"- Layers: {model.config.n_layer}")
logging.info(f"- Heads: {model.config.n_head}")
logging.info(f"- Embedding Dimension: {model.config.n_embd}")
logging.info("=" * 50)

# Log training hyperparameters
logging.info("Training Hyperparameters:")
logging.info(f"- Batch Size: {batch_size}")
logging.info(f"- Sequence Length: {seq_length}")
logging.info(f"- Gradient Accumulation Steps: {grad_accum_steps}")
logging.info(f"- Effective Batch Size: {effective_batch}")
logging.info(f"- Initial Learning Rate: {initial_lr}")
logging.info(f"- Minimum Learning Rate: {min_lr}")
logging.info(f"- Weight Decay: {0.1}")
logging.info(f"- Max Steps: {max_steps}")
logging.info(f"- Gradient Clip: {grad_clip}")
logging.info(f"- Target Loss: {target_loss}")
logging.info(f"- Early Stopping Patience: {max_patience}")
logging.info("=" * 50)

# Update .gitignore to keep logs