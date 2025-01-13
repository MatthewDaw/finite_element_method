import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from torch_geometric.nn import GraphConv, TopKPooling,  GCNConv, avg_pool, TAGConv, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import tqdm
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.model_selection import KFold

# Set torch device
if(torch.cuda.is_available()):
    print("USING GPU")
    device = torch.device("cuda:0")
else:
    print("USING CPU")
    device = torch.device("cpu")

class NodeSettingNet(torch.nn.Module):
    def __init__(self, output_dim, conv_width=64, topk=0.5, initial_num_nodes=None):

        super(NodeSettingNet, self).__init__()
        self.conv_width = conv_width
        self.initial_num_nodes = initial_num_nodes
        self.conv1 =  SAGEConv(2, conv_width)
        self.pool1 = TopKPooling(conv_width, ratio=topk)
        self.conv2 =  SAGEConv(conv_width, conv_width)
        self.pool2 = TopKPooling(conv_width, ratio= topk)
        self.conv3 =  SAGEConv(conv_width, conv_width)
        self.pool3 = TopKPooling(conv_width, ratio=topk)
        self.conv4 =  GCNConv(conv_width, conv_width)
        self.pool4 = TopKPooling(conv_width, ratio=topk)
        self.conv5 =  GCNConv(conv_width, conv_width)
        self.pool5 = TopKPooling(conv_width, ratio=topk)
        self.conv6 =  GCNConv(conv_width, conv_width)
        self.pool6 = TopKPooling(conv_width, ratio=topk)
        self.lin1 = torch.nn.Linear(2*conv_width, 128)
        self.lin2 = torch.nn.Linear(128, 100)
        self.lin3 = torch.nn.Linear(100, output_dim)
        torch.manual_seed(0)
        self.reset()

    def reset(self):
        """Reset the weights of the network."""
        # SAGEConv layers
        def randomize_weights(module):
            """Randomize weights of each layer."""
            if isinstance(module, (SAGEConv)):
                nn.init.xavier_normal_(module.lin_l.weight, gain=0.9)
                nn.init.normal_(module.lin_l.bias)
                nn.init.xavier_normal_(module.lin_r.weight, gain=0.9)

            if isinstance(module, (GCNConv)):
                nn.init.xavier_normal_(module.lin.weight, gain=0.9)

            if isinstance(module, (torch.nn.Linear)):
                nn.init.xavier_normal_(module.weight, gain=0.9)
                nn.init.normal_(module.bias)

        # Apply the randomization function to all layers
        self.apply(randomize_weights)

    def set_num_nodes(self, initial_num_nodes):
        self.initial_num_nodes = initial_num_nodes
        self.conv1 =  SAGEConv(self.initial_num_nodes, self.conv_width).to(device)

    def set_removable(self, removable):
        self.removable = removable

    def forward(self, data, embedding=False):
        """
        data: Batch of Pytorch Geometric data objects, containing node features, edge indices and batch size
            
        returns: Predicted normalized drag value
        """
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        #print(x.shape)

        x = F.relu(self.conv1(x, edge_index))
        #print(x.shape)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        #print(x.shape)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        #print(x.shape)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        #print(x.shape)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        #x = F.relu(self.conv3(x, edge_index))
        #print(x.shape)
        #x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        #print(x.shape)
        #x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv4(x, edge_index))
        #print(x.shape)
        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)
        #print(x.shape)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv5(x, edge_index))
        #print(x.shape)
        x, edge_index, _, batch, _, _ = self.pool5(x, edge_index, None, batch)
        #print(x.shape)
        x5 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        #x = F.relu(self.conv6(x, edge_index))
        #print(x.shape)
        #x, edge_index, _, batch, _, _ = self.pool6(x, edge_index, None, batch)
        #print(x.shape)
        #x6 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        #print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape, x6.shape)
        #raise

        #x = x1+x2+x3+x4+x5+x6
        x = x1+x2+x4+x5

        if(embedding):
            return x

        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.0, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        # pick which action to run
        x[:,:4] = F.softmax(x[:,:4], dim=1)

        # normalize add point outputs
        x[:,4:6] = x[:,4:6]

        # normalize remove point outputs
        x[:,6:8] = x[:,6:8]

        return x



class CriticNet(torch.nn.Module):
    def __init__(self, output_dim, conv_width=64, topk=0.5, initial_num_nodes=None):
        super(CriticNet, self).__init__()
        self.output_dim = output_dim
        self.conv_width = conv_width
        self.initial_num_nodes = initial_num_nodes
        self.conv1 = SAGEConv(2, conv_width)
        self.pool1 = TopKPooling(conv_width, ratio=topk)
        self.conv2 = SAGEConv(conv_width, conv_width)
        self.pool2 = TopKPooling(conv_width, ratio=topk)
        self.conv3 = GCNConv(conv_width, conv_width)
        self.pool3 = TopKPooling(conv_width, ratio=topk)
        self.conv4 = GCNConv(conv_width, conv_width)
        self.pool4 = TopKPooling(conv_width, ratio=topk)
        self.lin1 = torch.nn.Linear(2 * conv_width + self.output_dim, 128)
        self.lin2 = torch.nn.Linear(128, 100)
        self.lin3 = torch.nn.Linear(100, 1)

        torch.manual_seed(0)
        self.reset()

    def set_num_nodes(self, initial_num_nodes):
        self.initial_num_nodes = initial_num_nodes
        self.conv1 =  SAGEConv(self.initial_num_nodes, self.conv_width).to(device)


    def reset(self):
        """Reset the weights of the network."""
        def randomize_weights(module):
            """Randomize weights of each layer."""
            if isinstance(module, SAGEConv):
                nn.init.xavier_normal_(module.lin_l.weight, gain=0.9)
                nn.init.normal_(module.lin_l.bias)
                nn.init.xavier_normal_(module.lin_r.weight, gain=0.9)
            if isinstance(module, GCNConv):
                nn.init.xavier_normal_(module.lin.weight, gain=0.9)
            if isinstance(module, torch.nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.9)
                nn.init.normal_(module.bias)

        self.apply(randomize_weights)

    def forward(self, state_data, action_output, embedding=False):
        """
        Forward pass for the Critic network.
        state_data: Batch of state data (Pytorch Geometric data objects).
        action_output: Output of the actor network (tensor with actions).
        embedding: If True, return the embedding without Q-value prediction.

        Returns:
            Q-value (scalar) for the state-action pair.
        """
        # Extract node features, edge index, and batch from state data
        x, edge_index, batch = state_data.x.float(), state_data.edge_index, state_data.batch

        # Process the state through the convolutional and pooling layers
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv4(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Combine the features from the state and action output
        combined_features = torch.cat([x1+x2+x3+x4, action_output], dim=1)

        # Pass through fully connected layers
        x = F.relu(self.lin1(combined_features))
        x = F.relu(self.lin2(x))
        q_value = self.lin3(x)  # Q-value output

        return q_value
