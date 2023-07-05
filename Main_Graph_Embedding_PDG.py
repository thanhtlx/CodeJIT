import argparse
import os
from os import listdir
from os.path import isfile, join
import pandas
from graph_embedding.relational_graph_pdg import *
import sys
import pandas as pd

def is_stmt(string):
    if not isinstance(string,str):
        return False
    keys = ["methodDecl","ifStmt", "whileStmt", "return return" , "funcCall","assignment"]
    for k in keys:
        if string.startswith(k):
            return True
    return False

def convert_graph(node_info, edge_info):
    relation_map = dict()
    old_lines = set(node_info['lineNumber'])
    # select row represent line => 1 line, 1 node
    for _,row in node_info.iterrows():
        content = row['node_content']
        row['node_content'] = row['code']
        if not isinstance(content,str):
            continue
        if is_stmt(row['node_content']):
            relation_map[row['lineNumber']] = row
        if row['lineNumber'] in relation_map.keys():
            continue
        if isinstance(row['name'],str) and "<operator>." in row['name']:
            relation_map[row['lineNumber']] = row
        if row['lineNumber'] in relation_map.keys():
            continue
        if content.startswith('CtrlStruct'):
            relation_map[row['lineNumber']] = row
        if row['lineNumber'] in relation_map.keys():
            continue
        relation_map[row['lineNumber']] = row
    # check 
    # print(len(old_lines))
    # print(len(relation_map.keys()))
    assert len(old_lines) == len(relation_map.keys())
    map_index = dict()
    # map index to merge edge
    for k,v in relation_map.items():
        for _,row in  node_info[node_info['lineNumber'] == k].iterrows():
            map_index[row['id']] = v['id']
    # map edges 
    edges_list = list()
    for _,row in edge_info.iterrows():
        if 'innode' not in row or "outnode" not in row:
            continue
        if row['innode'] in map_index.keys():
            row['innode'] = map_index[row['innode']]
        if row['outnode'] in map_index.keys():
            row['outnode'] = map_index[row['outnode']]
        edges_list.append(row)
    node_info = pd.DataFrame(relation_map.values())
    edge_info = pd.DataFrame(edges_list)
    if edge_info.shape[0] > 0 :
        edge_info = edge_info[edge_info['outnode'].notna()]
        edge_info = edge_info[edge_info['innode'].notna()]
        if edge_info.shape[0] > 0:
            edge_info = edge_info.drop_duplicates(subset=['innode','outnode'])
    return node_info, edge_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_graph_dir', type=str, help='dir of the node files')
    parser.add_argument('--edge_graph_dir', type=str, help='dir of the edge files')
    parser.add_argument('--embedding_graph_dir', type=str, help='dir to save graph')
    parser.add_argument('--label', type=int, help='label of the commits, 1 if the commits are buggy, 0 otherwise')
    args = parser.parse_args()

    node_graph_dir = args.node_graph_dir
    edge_graph_dir = args.edge_graph_dir
    embedding_graph_dir = args.embedding_graph_dir
    if not os.path.isdir(embedding_graph_dir):
        os.makedirs(embedding_graph_dir)
    label = int(args.label)
    if label == 1:
        embedding_graph_dir = os.path.join(embedding_graph_dir, "VTC")
    else:
        embedding_graph_dir = os.path.join(embedding_graph_dir, "VFC")
    if not os.path.isdir(embedding_graph_dir):
        os.makedirs(embedding_graph_dir)
    node_files = [f for f in listdir(node_graph_dir) if isfile(join(node_graph_dir, f))]
    # random.shuffle(node_files)
    cm = set([f.split(".")[0].split("_")[-1] for f in node_files])
    #parsed = list()
    #with open('cm.txt') as f:
    #    parsed = set([l.strip() for l in f.readlines()])
    # parser_cm = list(cm.intersection(parsed))
    # with open('tmp') as f:
    #     parsed = set([l.strip() for l in f.readlines()])
    # parser_cm = list(set(parser_cm).difference(parsed))
    for commit_id in cm:
    # for commit_id in cm:
        try:
            # print(f"embedd: {commit_id}")
            save_path = os.path.join(embedding_graph_dir, "data_{}.pt".format(commit_id))
            print(save_path)
            if os.path.exists(save_path):
                print(f"embedded : {commit_id}")
                continue
            node_info = pandas.read_csv(join(node_graph_dir, "node_" + commit_id + ".csv"))
            edge_info = pandas.read_csv(join(edge_graph_dir, "edge_" + commit_id + ".csv"))
            # print(edge_info.shape)
            edge_info = edge_info[edge_info["etype"].isin(["CDG","DDG"])]
            print(commit_id)
            # print(node_info.head(1))
            node_info, edge_info =  convert_graph(node_info, edge_info)
            # print(edge_info.shape)
            # print(node_info)
            embed_graph(commit_id, label, node_info, edge_info, save_path)
        except Exception as e:
            exception_type, exception_object, exception_traceback = sys.exc_info()
            filename = exception_traceback.tb_frame.f_code.co_filename
            line_number = exception_traceback.tb_lineno
            print("Exception type: ", exception_type)
            print("File name: ", filename)
            print("Line number: ", line_number)
            print("exception:" + commit_id)
            print(e)