import argparse
import os
from os import listdir
from os.path import isfile, join
import torch
import pandas
from graph_embedding.relational_graph import *
import sys

import multiprocessing as mp

pool = mp.Pool(mp.cpu_count())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_graph_dir', type=str, help='dir of the node files')
    parser.add_argument('--edge_graph_dir', type=str, help='dir of the edge files')
    parser.add_argument('--embedding_graph_dir', type=str, help='dir to save embedding graph')
    parser.add_argument('--label', type=int, help='label of the commits, 1 if the commits are buggy, 0 otherwise')
    args = parser.parse_args()

    node_graph_dir = args.node_graph_dir
    edge_graph_dir = args.edge_graph_dir
    embedding_graph_dir = args.embedding_graph_dir
    if not os.path.isdir(embedding_graph_dir):
        os.makedirs(embedding_graph_dir)
    node_files = [f for f in listdir(node_graph_dir) if isfile(join(node_graph_dir, f))]
    label = int(args.label)
    if label == 1:
        embedding_graph_dir = os.path.join(embedding_graph_dir, "VTC")
    elif label == 2:
        embedding_graph_dir = os.path.join(embedding_graph_dir, "VUC")

    else:
        embedding_graph_dir = os.path.join(embedding_graph_dir, "VFC")
    if not os.path.isdir(embedding_graph_dir):
        os.makedirs(embedding_graph_dir)
    # random.shuffle(node_files)
    cm = set([f.split(".")[0].split("_")[-1] for f in node_files])
    parsed = list()
    #with open('cm.txt') as f:
    #    parsed = set([l.strip() for l in f.readlines()])
    #parser_cm = list(cm.intersection(parsed))
    for commit_id in cm:
        try:
            # commit_id = f.split(".")[0].split("_")[-1]
            save_path = os.path.join(embedding_graph_dir, "data_{}.pt".format(commit_id))
            if os.path.exists(save_path):
                print(f"embedded : {commit_id}")
                continue
            # node_info = pandas.read_csv(join(node_graph_dir, f))
            node_info = pandas.read_csv(join(node_graph_dir, "node_" + commit_id + ".csv"))
            edge_info = pandas.read_csv(join(edge_graph_dir, "edge_" + commit_id + ".csv"))
            edge_info = edge_info[edge_info["etype"] == "AST"]
            tmp = node_info.shape[0]
            add_lines = set(node_info[node_info['ALPHA']=='ADD']['lineNumber'])
            del_lines = set(node_info[node_info['ALPHA']=='DELETE']['lineNumber'])
            change_lines = add_lines.union(del_lines)
            node_info = node_info[node_info['lineNumber'].isin(change_lines)]
            print('node: ',tmp - node_info.shape[0], node_info.shape)
            node_list = list(node_info['id'])
            tmp = edge_info.shape[0]
            edge_info = edge_info[edge_info["outnode"].isin(node_list)]
            edge_info = edge_info[edge_info["innode"].isin(node_list)]
            print('edge: ',tmp - edge_info.shape[0], edge_info.shape)
            print("*"*33)
            
            #data = 
            #embed_graph(commit_id, label, node_info,  edge_info,save_path)
            pool.apply_async(embed_graph,args=(commit_id, label, node_info,  edge_info,save_path))
            # torch.save(data, save_path)
        except Exception as e:
            exception_type, exception_object, exception_traceback = sys.exc_info()
            filename = exception_traceback.tb_frame.f_code.co_filename
            line_number = exception_traceback.tb_lineno

            print("Exception type: ", exception_type)
            print("File name: ", filename)
            print("Line number: ", line_number)
            print("exception:" + commit_id)
            print(e)
    pool.close()
    pool.join()
