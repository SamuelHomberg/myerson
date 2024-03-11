import torch
from torch_geometric.nn import GATConv, global_add_pool
from torch.nn import functional as F
from torch.nn import Linear

class GATConvModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=256, *args, **kwargs):
        super().__init__()
        self.pool = global_add_pool
        self.conv1 = GATConv(dim_in, dim_hidden)
        self.conv2 = GATConv(dim_hidden, dim_hidden)
        self.conv3 = GATConv(dim_hidden, dim_hidden)
        self.fc1 = Linear(dim_hidden, dim_out)

    def forward(self, x, edge_index, batch, *args, **kwargs):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.pool(x, batch)
        x = self.fc1(x)
        
        return x