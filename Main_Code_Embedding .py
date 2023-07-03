import argparse
import os
from os import listdir
from os.path import isfile, join
import torch
import pandas
from graph_embedding.relational_graph import *
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
model = RobertaModel.from_pretrained("microsoft/codebert-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

import json


if __name__ == '__main__':
    embedding_graph_dir = "code"
    if not os.path.isdir(embedding_graph_dir):
        os.makedirs(embedding_graph_dir)
    with open('map_ids_diff_cm.json') as f:
        mm = json.load(f)
    for k,v in mm.items():
        save_file = join(embedding_graph_dir,f'data_{k}_.pt')
        _,data = model(torch.tensor(v,device=device)[None,:],return_dict=False)
        print(data.shape)
        torch.save(data,save_file)
    