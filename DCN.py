import torch
import torch.nn as nn
import torch.optim as optim

class CrossLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, 1))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x_0, x_l):
        x_l_proj = x_l @ self.weight
        x_cross = x_0 * x_l_proj + self.bias + x_l
        return x_cross
    
class DCN(nn.Module):
    def __init__(self, input_dim, cross_layers=3, hidden_units=[512, 256]):
        super().__init__()
        self.cross_network = nn.ModuleList([CrossLayer(input_dim) for _ in range(cross_layers)])

        deep_layers = []
        hidden_dim = input_dim
        for units in hidden_units:
            deep_layers.append(nn.Linear(hidden_dim, units))
            deep_layers.append(nn.ReLU())
            hidden_dim = units
        self.deep_network = nn.Sequential(*deep_layers)
        
        final_dim = input_dim + hidden_dim
        self.output_layer = nn.Linear(final_dim, 1)

    def forward(self, x):
        x_0, x_cross = x, x
        for layer in self.cross_network:
            x_cross = layer(x_0, x_cross)
        
        x_deep = self.deep_network(x)
        x_concat = torch.cat([x_cross, x_deep], dim=-1)
        out = self.output_layer(x_concat)
        # return torch.sigmoid(out)
        return out.squeeze(-1)


# Example usage
input_dim = 10
cross_layers = 3

# Initialize the DCN model
model = DCN(input_dim=input_dim, cross_layers=cross_layers)

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