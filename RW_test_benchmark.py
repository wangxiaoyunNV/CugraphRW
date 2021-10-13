"""
Random Walk test on Benchmark Datasets
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



# Data reader - the file format is MTX, so we will use the reader from SciPy
def read_and_create(datafile):
    # print('Reading ' + str(datafile) + '...')
    M = mmread(datafile).asfptype()

    _gdf = cudf.DataFrame()
    _gdf['src'] = M.row
    _gdf['dst'] = M.col
    _gdf['wt'] = 1.0

    _g = cugraph.Graph()
    _g.from_cudf_edgelist(_gdf, source='src', destination='dst', edge_attr='wt', renumber=False)

    # print("\t{:,} nodes, {:,} edges".format(_g.number_of_nodes(), _g.number_of_edges() ))

    return _g

def run_rw(_G, _seeds, _depth):
    t1 = time.time()
    _rw = cugraph.random_walks(_G, _seeds, _depth+1)
    # print(_rw)
    t2 = time.time() - t1
    return t2


import networkx as nx
import dgl
from dgl.sampling import random_walk, pack_traces
import torch as th

def read_dgl(datafile):
    # print('Reading ' + str(datafile) + '...')
    M = mmread(datafile).asfptype()
    # nx_g = nx.Graph(M)
    # _g = dgl.from_networkx(nx_g)
    src_ids = th.tensor(M.row)
    dst_ids = th.tensor(M.col)
    _g = dgl.graph((src_ids, dst_ids),idtype=th.int32)
    return _g

def run_dgl_rw(_G, _seeds, _depth):
    t1 = time.time()
    traces, types = random_walk(_G, nodes=_seeds, length=_depth)
    t2 = time.time() - t1
    return t2

data = ['coPapersCiteseer', 'as-Skitter']

for file in data:
    # dgl RW
    G_dgl = read_dgl('./data/'+ file + '.mtx')
    # cugraph RW
    G_cu = read_and_create('./data/'+ file + '.mtx')
    nodes = G_cu.nodes().to_array().tolist()

    # some parameters
    num_seeds_ = [1000, 3000, 5000, 10000, 20000, 40000, 75000, 100000]
    max_depth_ = np.arange(2,2**7+1,2)
    for max_depth in max_depth_:
        for num_seeds in num_seeds_:
            print('number of seeds:', num_seeds)
            print('RW length:', max_depth)

            # dgl RW
            # G_dgl = read_dgl('./data/'+ file + '.mtx')
            t_dgl = []
            for i in range(11):

                seeds = th.randint(0, G_dgl.num_nodes(), (num_seeds, ), dtype=th.int32)
                t = run_dgl_rw(G_dgl, seeds, max_depth)
                t_dgl.append(t)
                # print('dgl RW runtime: ',t)
                # print(t)
                # del G_dgl
            df_t_dgl = pd.DataFrame([t_dgl])
            df_t_dgl.to_csv('./RW_dgl_' + file + '_' + str(num_seeds) + '_.csv', mode='a', index=False, header=None)

            print(' ')

            # # cugraph RW
            # G_cu = read_and_create('./data/'+ file + '.mtx')
            # # num_nodes = G.number_of_nodes()
            # nodes = G_cu.nodes().to_array().tolist()
            t_cugraph = []
            for i in range(11):
                seeds = random.sample(nodes, num_seeds)

                t = run_rw(G_cu, seeds, max_depth)
                t_cugraph.append(t)
                # print('cugraph RW runtime: ',t)
                # print(t)
                # del G
            df_t_cugraph = pd.DataFrame([t_cugraph])
            df_t_cugraph.to_csv('./RW_cugraph_' + file + '_' + str(num_seeds) + '_.csv', mode='a', index=False, header=None)
