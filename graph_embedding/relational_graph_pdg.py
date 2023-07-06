
import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
change_operations = ["ADD", "DELETE", "REMAIN"]
edge_types = ["AST", "CFG", "CDG", "DDG"]


def load_nodes(nodes, index_col, encoders=None, **kwargs):
    if nodes.shape[0] <= 0:
        return torch.zeros(10,771), dict()

    mapping = {index: i for i, index in enumerate(nodes["id"].unique())}
    x = None
    if encoders is not None:
        xs = [encoder(nodes[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)
    return x, mapping


def load_edges(edges, src_index_col, src_mapping, dst_index_col, dst_mapping, edge_type_col,
               encoders=None, **kwargs):
    if edges.shape[0] <= 0:
        edge_index = torch.zeros(10,2)
        edge_type = torch.zeros(10)
        edge_attr = torch.zeros(10,3)
        # print(edge_index.shape)
        # print(edge_attr.shape)
        return edge_index, edge_type, edge_attr
    src = [src_mapping[index] for index in edges[src_index_col]]
    dst = [dst_mapping[index] for index in edges[dst_index_col]]

    edge_index = torch.tensor([src, dst])
    types = [edge_types.index(edge) for edge in edges[edge_type_col]]

    edge_type = torch.tensor(types)
    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(edges[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)
    return edge_index, edge_type, edge_attr

class ContentEncoder(object):
    def __init__(self, device=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        # print(device)
        self.caches = dict()
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.model.to(self.device)


    @torch.no_grad()
    def __call__(self, df):
        df = df.fillna("")
        x = torch.zeros(len(df), 768)
        for i, item in enumerate(df.values):
            item = item.strip()
            if len(item) <= 0:
                continue
            tokens = self.tokenizer.tokenize(item)[:500]
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings  =  self.model(torch.tensor(tokens_ids,device=self.device)[None,:])[0][0]
            context_embeddings = context_embeddings.clone().detach()
            if len(context_embeddings) > 0:
                x[i] =  torch.sum(context_embeddings, dim=0)
        return x



class OneHotEncoder(object):
    def __init__(self, dicts):
        self.dicts = dicts

    def __call__(self, df):
        x = torch.zeros(len(df), len(self.dicts))
        for i, col in enumerate(df.values):
            x[i, self.dicts.index(col)] = 1
        return x



def embed_graph(commit_id, ground_truth, nodes, edges,save_path):
    node_x, node_mapping = load_nodes(
        nodes, index_col='id', encoders={
            'ALPHA': OneHotEncoder(change_operations),
            'node_content': ContentEncoder()
        })


    edge_index, edge_type, edge_label = load_edges(
        edges,
        src_index_col='outnode',
        src_mapping=node_mapping,
        dst_index_col='innode',
        dst_mapping=node_mapping,
        edge_type_col = 'etype',
        encoders={'change_operation': OneHotEncoder(change_operations)}
    )


    data = Data()
    data.x = node_x
    data.edge_index = edge_index
    data.edge_attr = edge_label
    data.edge_type = edge_type

    data.y = torch.tensor([ground_truth], dtype = int)
    torch.save(data, save_path)
    return data


def get_node_mapping(nodes, edges):

    mapping = {index: i for i, index in enumerate(nodes["id"].unique())}
    return mapping

def load_edge_mapping(edges, src_index_col, src_mapping, dst_index_col, dst_mapping):
    #df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in edges[src_index_col]]
    dst = [dst_mapping[index] for index in edges[dst_index_col]]

    edge_index = torch.tensor([src, dst])
    return edge_index