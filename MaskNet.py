import torch
import torch.nn as nn

class MaskBlock(nn.Module):
	def __init__(self, input_dim):
		super().__init__()
		self.mask_mlp = nn.Sequential(
			nn.Linear(input_dim, input_dim),
			nn.Sigmoid()
		)
	def forward(self, x):
		mask = self.mask_mlp(x)
		return mask * x

class MaskNet:
	def __init__(self, sparse_field_dims, dense_dim, embed_dim, hidden_dims=[128, 64], dropout=0.1):

		self.sparse_embeddings = nn.ModuleList([
			nn.Embedding(field_size, embed_dim) for field_size in sparse_field_dims
		])
		self.input_dim = len(sparse_field_dims) * embed_dim + dense_dim
		dims = [self.input_dim] + hidden_dims

		self.blocks = nn.ModuleList()
		for i in range(len(hidden_dims)):
			self.blocks.append(nn.Sequential(
				MaskBlock(dims[i]),
				nn.Linear(dims[i], dims[i+1]),
				nn.ReLU(),
				nn.Dropout(dropout)
			))

		self.output_layer = nn.Linear(hidden_dims[-1], 1)

	def forward(self, sparse_inputs, dense_inputs):
		embed_outs = [
			emb(sparse_inputs[:, i]) for i, emb in enumerate(self.sparse_embeddings)
		]
		embed_concat = torch.cat(embed_outs, dim=1)
		x = torch.cat([embed_concat, dense_inputs], dim=1)

		for block in self.blocks:
			x = block(x)
		out = self.output_layer(x)
		return torch.sigmoid(out)

if __name__ == "__main__":
	sparse_field_dims = [10000, 10000, 10000, 10000]
	dense_dim = 10
	embed_dim = 8
	hidden_dims = [128, 64]
	dropout = 0.1

	model = MaskNet(sparse_field_dims, dense_dim, embed_dim, hidden_dims, dropout)
	print(model.blocks)
	