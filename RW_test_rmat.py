"""
Random Walk test on Graphs by cuGraph.generator.rmat
"""
import numpy as np
import pandas as pd
#  Import the modules
import cugraph
import cudf
from cugraph.generators import rmat
import networkx as nx
from sklearn import preprocessing

# system and other
import gc
import os
import time
import random

# MTX file reader
from scipy.io import mmread

def generate_graph(_scale):
    cu_df = cugraph.generators.rmat(_scale, (2**_scale)*16, 0.1,0.2, 0.3, 42, clip_and_flip=False, scramble_vertex_ids=True, create_using=None, mg=False)
    # df = cu_df.to_pandas()
    # G_nx = nx.from_pandas_edgelist(df,'src','dst')
    # G_nx_ = nx.convert_node_labels_to_integers(G_nx)
    # df_ =nx.to_pandas_edgelist(G_nx_)
    le = preprocessing.LabelEncoder()
    le.fit(np.unique(cu_df.values).tolist())
    df_ = pd.DataFrame()
    df_['src'] = le.transform(cu_df['src'].to_array().tolist()).astype('int32')
    df_['dst'] = le.transform(cu_df['dst'].to_array().tolist()).astype('int32')
    return df_


def generate_cugraph(df):
    _G = cugraph.Graph()
    _G.from_pandas_edgelist(df, source='src', destination='dst', edge_attr=None, renumber=False)
    _G.edges()
    return _G


def run_rw(_G, _seeds, _depth):
    t1 = time.time()
    _rw = cugraph.random_walks(_G, _seeds, _depth+1)
    # print(_rw)
    t2 = time.time() - t1
    return t2


import dgl
from dgl.sampling import random_walk, pack_traces
import torch as th

def create_dgl(df):
    src_ids = th.tensor(df['src'])
    dst_ids = th.tensor(df['dst'])
    _g = dgl.graph((src_ids, dst_ids),idtype=th.int32)
    return _g


def run_dgl_rw(_G, _seeds, _depth):
    t1 = time.time()
    traces, types = random_walk(_G, nodes=_seeds, length=_depth)
    t2 = time.time() - t1
    return t2


scale_ = [18, 19, 20, 21]
# scale_ = [5]

for scale in scale_:
    df = generate_graph(scale)

    # cugraph
    G_cu = generate_cugraph(df)
    # num_nodes = G.number_of_nodes()
    nodes = G_cu.nodes().to_array().tolist()

    # dgl
    G_dgl = create_dgl(df)

    # some parameters
    num_seeds_ = [1000,3000,5000,10000,20000,40000,75000,100000]
    # num_seeds_ = [10]
    max_depth_ = np.arange(2,2**7+1,2)
    # max_depth_ = np.arange(2,5,2)
    for max_depth in max_depth_:
        for num_seeds in num_seeds_:
            print('scale', scale)
            print('number of seeds:', num_seeds)
            print('RW length:', max_depth)

            # # cugraph RW
            # G_cu = generate_cugraph(df)
            # # num_nodes = G.number_of_nodes()
            # nodes = G_cu.nodes().to_array().tolist()
            t_cugraph = []
            for i in range(11):
                seeds = random.sample(nodes, num_seeds)
                # seeds = random.choices(nodes, k=num_seeds)

                t = run_rw(G_cu, seeds, max_depth)
                t_cugraph.append(t)
                # print('cugraph RW runtime: ',t)
                # print(t)
                # del G
            df_t_cugraph = pd.DataFrame([t_cugraph])
            df_t_cugraph.to_csv('./RW_cugraph_' + str(scale) + '_' + str(num_seeds) + '_.csv', mode='a', index=False, header=None)

            print(' ')

            # dgl RW
            # G_dgl = create_dgl(df)
            t_dgl = []
            for i in range(11):

                seeds = th.randint(0, G_dgl.num_nodes(), (num_seeds, ), dtype=th.int32)
                # seeds = random.choices(nodes, num_seeds)
                t = run_dgl_rw(G_dgl, seeds, max_depth)
                t_dgl.append(t)
                # print('dgl RW runtime: ',t)
                # print(t)
                # del G_dgl
            df_t_dgl = pd.DataFrame([t_dgl])
            df_t_dgl.to_csv('./RW_dgl_' + str(scale) + '_' + str(num_seeds) + '_.csv', mode='a', index=False, header=None)
