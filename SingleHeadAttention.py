import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)

        # linear projections for query, key, value
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # optional output projection
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def foward(self, x, mask=None):
        """
        :param x: Tensor of shape [batch_size, seq_len, embed_dim]
        :param mask:
        :return:
        """
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        d_k = self.embed_dim
        """
        It swaps the last two dimensions of a tensor.
        Typical shape of Q and K: [batch_size, num_heads, seq_len, head_dim]
        To compute Q @ Kᵀ, we need Kᵀ to be of shape [batch_size, num_heads, head_dim, seq_len]
        So we do .transpose(-1, -2) to flip seq_len and head_dim
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
        if mask is not None:
            # Softmax is used on scores. When a score is -inf, softmax gives it a probability of 0.
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax along last dimension (columns)
        # Attention scores: [batch, heads, query_len, key_len] → softmax over the keys (key_len, so dim=-1)
        # the key_len is same as seq_len
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        # Final projection
        output = self.out_linear(output)
        return output, attn_weights

if __name__ == '__main__':
    batch_size = 2
    seq_len = 5
    embed_dim = 16

    x = torch.randn(batch_size, seq_len, embed_dim)

    # Causal mask: allow attention only to current and previous tokens
    """
    Do you need to match the batch size?
    Usually not, because PyTorch supports broadcasting.
    When you apply this mask to attention scores of shape [batch_size, num_heads, seq_len, seq_len], 
    PyTorch can broadcast the [1, seq_len, seq_len] mask across batch_size and num_heads.
    """
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)  # (1, seq_len, seq_len)

    attn = SingleHeadAttention(embed_dim)
    out, weights = attn.forward(x, causal_mask)
