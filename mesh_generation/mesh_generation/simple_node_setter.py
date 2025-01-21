import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

class PointGeneratorGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, weight_clip_value=1.0):
        super(PointGeneratorGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weight_clip_value = weight_clip_value

        # Custom weight initialization
        self.init_weights()

    def init_weights(self):
        # GCNConv layers are initialized by default
        # Custom initialization for the final linear layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def clip_weights(self):
        # Clip weights of all layers
        for param in self.parameters():
            param.data.clamp_(-self.weight_clip_value, self.weight_clip_value)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = torch.relu(self.conv1(x.float(), edge_index))
        x = torch.relu(self.conv2(x, edge_index))

        x = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]

        return self.fc(x)  # [num_graphs, output_dim]

