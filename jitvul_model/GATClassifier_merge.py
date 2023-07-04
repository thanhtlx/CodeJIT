from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GAT
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import torch
from torch.nn import ReLU, Softmax, LeakyReLU


class GATClassifier(torch.nn.Module):
    def __init__(self, in_channels,in_channels2, hidden_channels, num_relations=4, dropout=0.1, num_of_layers=2, graph_readout_func="add"):
        super(GATClassifier, self).__init__()
        torch.manual_seed(12345)
        self.num_layers = num_of_layers
        self.num_of_relations = num_relations
        self.dropout = dropout
        self.graph_readout_func = graph_readout_func
        for i in range(self.num_layers):
            if i == 0:
                exec('self.conv_{} = GAT(in_channels, hidden_channels, num_layers = self.num_layers, dropout = dropout)'.format(i))
            else:
                exec('self.conv_{} = GAT(hidden_channels, hidden_channels, num_layers = self.num_layers, dropout = dropout)'.format(i))

        for i in range(self.num_layers):
            if i == 0:
                exec('self.conv2_{} = GAT(in_channels2, hidden_channels, num_layers = self.num_layers, dropout = dropout)'.format(i))
            else:
                exec('self.conv2_{} = GAT(hidden_channels, hidden_channels, num_layers = self.num_layers, dropout = dropout)'.format(i))
        self.relu = ReLU(inplace=True)
        self.lin = Linear(hidden_channels*2, 2)
        self.lin1 = Linear(hidden_channels, 2)
        self.out = Softmax(dim=0)

    def forward(self, x, edge_index1, edge_attr1, y, edge_index2, edge_attr2):
        # 1. Obtain node embeddings
        # print(x.shape,edge_index1.shape,edge_attr1.shape)
        # print(y.shape,edge_index2.shape,edge_attr2.shape)
        
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                x = eval(
                    'self.conv_{}(x, edge_index1, edge_attr = edge_attr1)'.format(i))
                x = self.relu(x)
            else:
                x = eval(
                    'self.conv_{}(x, edge_index1, edge_attr = edge_attr1)'.format(i))
        # 2. Readout layer
        if self.graph_readout_func == "average":
            x = global_mean_pool(x,  torch.zeros(
                x.shape[0], dtype=int, device=x.device))
        if self.graph_readout_func == "max":
            x = global_max_pool(x,  torch.zeros(
                x.shape[0], dtype=int, device=x.device))
        else:
            x = global_add_pool(x,  torch.zeros(
                x.shape[0], dtype=int, device=x.device))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                y = eval(
                    'self.conv2_{}(y, edge_index2, edge_attr = edge_attr2)'.format(i))
                y = self.relu(y)
            else:
                y = eval(
                    'self.conv2_{}(y, edge_index2, edge_attr = edge_attr2)'.format(i))
        # 2. Readout layer
        if self.graph_readout_func == "average":
            y = global_mean_pool(y,  torch.zeros(
                y.shape[0], dtype=int, device=y.device))
        if self.graph_readout_func == "max":
            y = global_max_pool(y,  torch.zeros(
                y.shape[0], dtype=int, device=y.device))
        else:
            y = global_add_pool(y,  torch.zeros(
                y.shape[0], dtype=int, device=y.device))
        y = F.dropout(y, p=self.dropout, training=self.training)
        
        # 3. Apply a final classifier
        # out = self.lin(out)
        out = torch.cat((x,y), 0)
        # print(out.shape)
        
        out,_ = torch.max(out,0)
        # print(out.shape)
        # print(out.shape)
        out = out.reshape(1,out.shape[0])
        out = self.lin1(out)
        return out
