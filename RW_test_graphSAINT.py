"""
Random Walk test on GraphSAINT Datasets
"""
import numpy as np
import pandas as pd
#  Import the modules
import cugraph
import cudf

# system and other
import gc
import os
import time
import random

# MTX file reader
from scipy.io import mmread
import scipy.sparse

import networkx as nx
import dgl
from dgl.sampling import random_walk, pack_traces
import torch as th



def read_and_create(datafile):
    adj_full = scipy.sparse.load_npz('./data/' + datafile + '/adj_full.npz')

    offsets = cudf.Series(adj_full.indptr)
    indices = cudf.Series(adj_full.indices)
    weights = cudf.Series(adj_full.data)

    _g = cugraph.Graph()
    _g.from_cudf_adjlist(offsets, indices, weights)
    _g.edges()
    return _g

def run_rw(_G, _seeds, _depth):
    t1 = time.time()
    _rw = cugraph.random_walks(_G, _seeds, _depth+1)
    # print(_rw)
    t2 = time.time() - t1
    return t2


def read_dgl(datafile):
    adj_full = scipy.sparse.load_npz('./data/' + datafile + '/adj_full.npz')
    _g = dgl.from_scipy(adj_full)
    # num_nodes = _g.num_nodes()
    return _g

def run_dgl_rw(_G, _seeds, _depth):
    t1 = time.time()
    traces, types = random_walk(_G, nodes=_seeds, length=_depth)
    t2 = time.time() - t1
    return t2

data = ['ppi', 'flickr', 'reddit', 'yelp', 'amazon']

for file in data:
    # some parameters
    num_seeds_ = [1000, 3000, 5000, 10000, 20000, 40000, 75000, 100000]
    max_depth_ = np.arange(2,2**7+1,2)

    # dgl RW
    G_dgl = read_dgl(file)

    # cugraph RW
    G_cu = read_and_create(file)
    # num_nodes = G.number_of_nodes()
    nodes = G_cu.nodes().to_array().tolist()

    for max_depth in max_depth_:
        for num_seeds in num_seeds_:
            print('number of seeds:', num_seeds)
            print('RW length:', max_depth)

            # # dgl RW
            # G_dgl = read_dgl(file)
            t_dgl = []
            for i in range(11):

                seeds = th.randint(0, G_dgl.num_nodes(), (num_seeds, ), dtype=th.int64)
                t = run_dgl_rw(G_dgl, seeds, max_depth)
                t_dgl.append(t)
                # print('dgl RW runtime: ',t)
                # print(t)
                # del G_dgl
            df_t_dgl = pd.DataFrame([t_dgl])
            df_t_dgl.to_csv('./RW_dgl_' + file + '_' + str(num_seeds) + '_.csv', mode='a', index=False, header=None)

            print(' ')

            # # cugraph RW
            # G_cu = read_and_create(file)
            # # num_nodes = G.number_of_nodes()
            # nodes = G_cu.nodes().to_array().tolist()
            t_cugraph = []
            for i in range(11):
                # seeds = random.sample(nodes, num_seeds)
                seeds = random.choices(nodes, k=num_seeds)

                t = run_rw(G_cu, seeds, max_depth)
                t_cugraph.append(t)
                # print('cugraph RW runtime: ',t)
                # print(t)
                # del G
            df_t_cugraph = pd.DataFrame([t_cugraph])
            df_t_cugraph.to_csv('./RW_cugraph_' + file + '_' + str(num_seeds) + '_.csv', mode='a', index=False, header=None)
