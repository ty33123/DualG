# -*- coding: utf-8 -*-
from tqdm import tqdm
import numpy as np
import torch
import os
import json


def load_tables(data_dir):
    with open(os.path.join(data_dir, 'tables.json')) as f:
        tables = json.load(f)
    return tables


def load_queries(data_dir, file_name='queries.txt'):
    queries = {}
    with open(os.path.join(data_dir, file_name)) as f:
        for line in f.readlines():
            query = line.strip().split()
            queries[query[0]] = query[1:]
    return queries


def load_qt_relations(data_dir):
    """
        数据格式为: 1	0	table-0370-614	2
        第二列是无用列。第一列为问题编号，第三列为表格编号，第四列为相关度(0,1,2)
        qtrels[问题编号][表格编号]=相关度
    """
    qtrels = {}
    with open(os.path.join(data_dir, 'qtrels.txt')) as f:
        for line in f.readlines():
            rel = line.strip().split()
            rel[0] = rel[0]
            rel[3] = int(rel[3])
            if rel[0] not in qtrels:
                qtrels[rel[0]] = {}
            qtrels[rel[0]][rel[2]] = rel[3]
    return qtrels


def process_tables(tables, constructor):
    """
        对表格进行处理,将表格,dgl全称:Deep graph Library
    """
    for tid in tqdm(tables.keys(), desc="processing tables"):
        row_graph, col_graph, node_features, table_data = constructor.construct_graph(
            tables[tid])
        tables[tid]["row_graph"] = row_graph
        tables[tid]["col_graph"] = col_graph
        if isinstance(node_features, np.ndarray):
            node_features = torch.FloatTensor(node_features)
        tables[tid]["node_features"] = node_features


def process_queries(queries, constructor):
    """
        对查询进行处理，将查询编码为FloatTensor。
    """
    features = {}
    for qid in tqdm(queries.keys(), desc="processing queries"):
        feature = constructor.w2v[" ".join(queries[qid])]
        if isinstance(feature, np.ndarray):
            feature = torch.FloatTensor(feature)
        features[qid] = feature
    return features
