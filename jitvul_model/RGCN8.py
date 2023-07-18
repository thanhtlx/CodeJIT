from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import (
    RGCNConv,
    RGATConv,
    CuGraphRGCNConv,
    GATConv,
    FastRGCNConv,
)
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import torch
from torch.nn import ReLU, Softmax, LeakyReLU

# upscale ctg and convert same range value


class RGCN8(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        edge_dim,
        num_relations=4,
        dropout=0.1,
        num_of_layers=2,
        graph_readout_func="add",
        code_embedding_size=768,
    ):
        super(RGCN8, self).__init__()
        torch.manual_seed(12345)
        self.num_layers = num_of_layers
        self.num_of_relations = num_relations
        self.dropout = dropout
        self.graph_readout_func = graph_readout_func
        for i in range(self.num_layers):
            if i == 0:
                exec(
                    "self.conv_{} = RGCNConv(in_channels, hidden_channels,num_relations=self.num_of_relations, add_self_loops=False, dropout = dropout)".format(
                        i
                    )
                )
            else:
                exec(
                    "self.conv_{} = RGCNConv(hidden_channels, hidden_channels,num_relations=self.num_of_relations, add_self_loops=False, dropout = dropout)".format(
                        i
                    )
                )
        self.relu = ReLU(inplace=True)
        self.lin = Linear(hidden_channels, 2)
        self.lin_ctg = Linear(hidden_channels, code_embedding_size)
        self.dan1 = Linear(code_embedding_size, code_embedding_size)
        self.dan2 = Linear(code_embedding_size, code_embedding_size)
        # self.merge = Linear(,hidden_channels)
        self.out = Linear(code_embedding_size, 2)

    def forward(self, x, edge_index, edge_type, edge_attr, embed, msg=None):
        # 1. Obtain node embeddings
        # for i in range(self.num_layers):
        #     if i < self.num_layers - 1:
        #         x = eval('self.conv_{}(x, edge_index, edge_type)'.format(i))
        #         x = self.relu(x)
        #     else:
        #         x = eval('self.conv_{}(x, edge_index, edge_type)'.format(i))
        # # 2. Readout layer
        # if self.graph_readout_func == "average":
        #     x = global_mean_pool(x,  torch.zeros(
        #         x.shape[0], dtype=int, device=x.device))
        # if self.graph_readout_func == "max":
        #     x = global_max_pool(x,  torch.zeros(
        #         x.shape[0], dtype=int, device=x.device))
        # else:
        #     x = global_add_pool(x,  torch.zeros(
        #         x.shape[0], dtype=int, device=x.device))
        #     # 3. Apply a final classifier
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # # codebert
        # x = self.lin_ctg(x)
        # x = self.relu(x)
        #
        #
        embed = self.dan1(embed)
        embed = self.dan2(embed)
        y = self.relu(embed)
        # z = self.relu(msg)
        # merge
        # merge = torch.cat((x, y), dim=1)
        # merge = self.merge(merge)
        out = self.out(y)
        return out
