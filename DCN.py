import torch
import torch.nn as nn
import torch.optim as optim

class DCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, cross_layers, output_dim=1):
        super().__init__()

        # deep network, simple MLP
        self.deep_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Cross network layers (Cross layers)
        self.cross_weights = nn.ParameterList([nn.Parameter(torch.randn(input_dim, 1)) for _ in range(cross_layers)])
        self.cross_biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(cross_layers)])
        self.cross_layers = cross_layers

        self.output_layer = nn.Linear(input_dim + hidden_dim, output_dim)

    def forward(self, x):
        x0 = x
        xl = x #(B, input_dim)
        for i in range(self.cross_layers):
            xw = xl @ self.cross_weights[i] + self.cross_biases[i] # (B, 1)
            print(f"xw shape: {xw.shape}")
            cross = x0 * xw  # element-wise multiplication with broadcast
            print(f"cross shape: {cross.shape}")
            xl = cross + xl
        cross_out = xl

        deep_out = self.deep_network(x)

        combined_out = torch.cat([cross_out, deep_out], dim=-1)
        out = self.output_layer(combined_out)
        return out.squeeze(-1) #(batch, )



# Example usage
input_dim = 10  # Number of features
hidden_dim = 64  # Hidden units in deep network
cross_layers = 3  # Number of cross layers

# Initialize the DCN model
model = DCN(input_dim=input_dim, hidden_dim=hidden_dim, cross_layers=cross_layers)

# Sample data: Let's create a random input tensor of size (batch_size, input_dim)
batch_size = 32
x = torch.randn(batch_size, input_dim)
y = torch.randn(batch_size)

# Forward pass
y_pred = model(x)
# print(y_pred.shape)

# Training step (regression)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = loss_fn(y_pred, y)
loss.backward()
optimizer.step()
print("Loss:", loss.item())