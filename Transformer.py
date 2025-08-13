import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention:
	def __init__(self, embed_dim, num_heads):
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.head_dim = embed_dim // num_heads

		self.qkv_proj = nn.Linear(embed_dim, 3*embd_dim)
		self.out_proj = nn.Linear(embed_dim, embed_dim)

	def forward(self, X):
		B, S, E = X.size()
		qkv = self.qkv_proj(x)

		# (3, batch_size, num_heads, sequence_length, embed_dim)
		qkv = qkv.view(B, S, 3, self.num_heads, self.embed_dim).permute(2, 0, 3, 1, 4)
		Q, K, V = qkv[0], qkv[1], qkv[2]

		scores = Q @ K.tranpose(-2, -1) / (self.head_dim ** 0.5)
		attn_weights = F.softmax(scores, dim=-1)
		out = attn_weights @ V

		# change to (batch_size, sequence_length, num_head, embed_dim)
		# embed_dim = num_head * head_dim
		out = out.transpose(1, 2).continguous().view(B, S, E)
		return self.out_linear(out)


class FeedForward(nn.Module):
	
	def __init__(self, embed_dim, hidden_dim):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(embed_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, embed_dim)
		)

	def forward(self, x):
		return self.net(x)

class PositionalEncoding(nn.Module):
	def __init__(self, embed_dim, max_len=5000):
		super().__init__()
		pe = torch.zeros(max_len, embed_dim)
		position = torch.arange(0, max_len).unsqueeze(1).float()
		div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch,tensor(10000.0)) / embed_dim))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:, :x.size(1)]
		return x


class TransformerBlock(nn.Module):
	def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
		super().__init__()
		self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
		self.ff = FeedForward(embed_dim, hidden_dim)
		self.norm1 = nn.LayerNorm(embed_dim)
		self.norm2 = nn.LayerNorm(embed_dim)
		self.dropout = nn.Dropout(dropout)


	def forward(self, x):
		x = x + self.dropout(self.attn(self.norm1(x)))
		x = x + self.dropout(self.ff(self.norm2(x)))
		return x
	

class TransformerEncoder(nn.Module):
	def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len=512):
		super().__init__()
		self.token_embed = nn.Embedding(vocab_size, embed_dim) # highest weights
		self.pos_embed = PositionalEncoding(embed_dim, max_len)
		self.layers = nn.ModuleList([
			TransformerBlock(embed_dim, num_heads, hidden_dim)
			for _ in range(num_layers)
		])
		self.norm = nn.LayerNorm(embed_dim)

	def forward(self, x):
		x = self.token_embed(x)
		x = self.pos_embed(x)
		for layer in self.layers:
			x = layer(x)
		return self.norm(x)
	

class MaskedMultiHeadSelfAttention(nn.Module):
	def __init__(self, embed_dim, num_heads):
		super().__init__()
		self.mha = MultiHeadSelfAttention(embed_dim, num_heads)
	
    def forward(self, x, mask=None):
        attn_output = self.mha(x)
        if mask is not None:
			attn_output = attn_output.masked_fill(mask == 0, float('-inf'))
		return attn_output



class TransformerDecoderBlock(nn.Module):
	def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
		super().__init__()
		self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
		self.cross_attn = MultiHeadSelfAttention(embed_dim, num_heads)
		self.ff = FeedForward(embed_dim, hidden_dim)
		self.norm1 = nn.LayerNorm(embed_dim)
		self.norm2 = nn.LayerNorm(embed_dim)
		self.norm3 = nn.LayerNorm(embed_dim)
		self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x)))
        x = x + self.dropout(self.cross_attn(self.norm2(x), encoder_output, encoder_output))
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x


class TransformerDecoder(nn.Module):
	def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len=512):
		super().__init__()
		self.token_embed = nn.Embedding(vocab_size, embed_dim)
		self.pos_embed = PositionalEncoding(embed_dim, max_len)
		self.layers = nn.ModuleList([
			TransformerDecoderBlock(embed_dim, num_heads, hidden_dim)
			for _ in range(num_layers)
		])
		self.norm = nn.LayerNorm(embed_dim)


	def forward(self, tgt, encoder_output, tgt_mask=None, memory_mask=None):
		x = self.token_embed(tgt)
		x = self.pos_embed(x)
		
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, memory_mask)
        return self.norm(x)


def generate_random_mask(seq_le):
	return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)


if __name__ == "__main__":
	encoder = TransformerEncoder(vocab_size=10000, embed_dim=512, num_heads=8, hidden_dim=2048, num_layers=6)
	decoder = TransformerDecoder(vocab_size=10000, embed_dim=512, num_heads=8, hidden_dim=2048, num_layers=6)
	
    # 2 is the batch size, 32 is the sequence length, 10000 is the vocabulary size
    src = torch.randint(0, 10000, (2, 32))
    tgt = torch.randint(0, 10000, (2, 16))

    encoder_output = encoder(src)
    tgt_mask = generate_random_mask(tgt.size(1))

    output = decoder(tgt, encoder_output, tgt_mask)