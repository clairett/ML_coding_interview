import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2


# -----------------
class Head(nn.Module):
	""" one head of self-attention """

	def __init__(self, head_size, n_embed=32, block_size=8):
		super().__init__()
		self.key = nn.Linear(n_embed, head_size, bias=False)
		self.query = nn.Linear(n_embed, head_size, bias=False)
		self.value = nn.Linear(n_embed, head_size, bias=False)
		self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		B, T, C = x.shape
		k = self.key(x)  # (B, T, C)
		q = self.query(x)  # (B, T, C)

		wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,T,C) ---> (B,T,T)
		wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
		wei = F.softmax(wei, dim=-1)
		wei = self.dropout(wei)

		v = self.value(x)
		out = wei @ v
		return out
	
	
class MultiHeadAttention(nn.Module):
	def __init__(self, num_heads, head_size):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
		self.proj = nn.Linear(num_heads * head_size, num_heads * head_size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = self.proj(out)
		return out
	

class FeedForward(nn.Module):
	
	def __init__(self, n_embed):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embed, 4 * n_embed),
			n.ReLU(),
			nn.Linear(4 * n_embed, n_embed),
			nn.Dropout(dropout)
		)

	def forward(self, x):
		return self.net(x)
		

class Block(nn.Module):
	""" Transformer blockL communication followed by computation """

	def __init__(self, n_embed, n_head):
		super().__init__()
		head_size = n_embed // n_head
		self.sa = MultiHeadAttention(n_head, head_size)
		self.ffwd = FeedForward(n_embed)
		self.ln1 = nn.LayerNorm(n_embed)
		self.ln2 = nn.LayerNorm(n_embed)

	def forward(self, x):
		x = x + self.sa(self.ln1(x))
		x = x + self.ffwd(self.ln2(x))
		return x

	


