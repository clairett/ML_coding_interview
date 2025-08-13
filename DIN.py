import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionUnit(nn.Module):
	def __init__(self, embed_dim, att_hidden_units):
		super().__init__()
		self.fc1 = nn.Linear(embed_dim * 2, att_hidden_units)
		self.fc2 = nn.Linear(att_hidden_units, 1)

	def forward(self, behavior_emb, candidate_emb):
		# behavior_emb: (B, S, E), candidate_emb: (B, E)
		B, S, E = behavior_emb.shape
		candidate_expanded = candidate_emb.unsqueeze(1).expand(-1, S, -1) # (B, S, E)
		concat = torch.cat([behavior_emb, candidate_expanded], dim=-1) # (B, S, 2E)
		attn = F.relu(self.fc1(concat))
		attn = self.fc2(attn).squeeze(-1)  # (B, S)
		attn = F.softmax(attn, dim=-1)
		output = (behavior_emb * attn.unsqueeze(-1)).sum(dim=1)
		return output

class DIN(nn.Module):
	def __init__(self, embed_dim, att_hidden_units, mlp_hidden_units):
		super().__init__()
		self.attention = AttentionUnit(embed_dim, att_hidden_units)
		self.mlp = nn.Sequential(
			nn.Linear(embed_dim * 2, mlp_hidden_units),
			nn.ReLU(),
			nn.Linear(mlp_hidden_units, 1)
		)
	def forward(self, behavior_emb, candidate_emb, user_features=None, item_features=None):
		user_interest = self.attention(behavior_emb, candidate_emb)
		# both have (B, E), we get (B, 2*E)
		x = torch.cat([user_interest, candidate_emb], dim=1)
		logit = self.mlp(x).squeeze(-1)
		return logit
	
