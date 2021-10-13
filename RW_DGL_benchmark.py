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
import networkx as nx
import dgl
from dgl.sampling import random_walk, pack_traces
import torch as th

def read_dgl(datafile):
    M = mmread(datafile).asfptype()
    src_ids = th.tensor(M.row)
    dst_ids = th.tensor(M.col)
    _g = dgl.graph((src_ids, dst_ids),idtype=th.int32)
    return _g

def run_dgl_rw(_G, _seeds, _depth):
    t1 = time.time()
    traces, types = random_walk(_G, nodes=_seeds, length=_depth)
    t2 = time.time() - t1
    return t2

data = ['preferentialAttachment', 'dblp-2010', 'as-Skitter', 'citationCiteseer', 'caidaRouterLevel', 'coAuthorsDBLP', 'coPapersDBLP', 'coPapersCiteseer']

for file_name in data:
    # dgl RW
    G_dgl = read_dgl('./data/'+ file_name + '.mtx')
    # some parameters
    num_seeds_ = [1000, 3000, 5000, 10000, 20000, 40000, 75000, 100000]
    max_depth_ = np.arange(2,2**7+1,2)
    for max_depth in max_depth_:
        for num_seeds in num_seeds_:
            print('number of seeds:', num_seeds)
            print('RW length:', max_depth)
            t_dgl = []
            for i in range(11):
                seeds = th.randint(0, G_dgl.num_nodes(), (num_seeds, ), dtype=th.int32)
                t = run_dgl_rw(G_dgl, seeds, max_depth)
                t_dgl.append(t)
                
            df_t_dgl = pd.DataFrame([t_dgl])
            df_t_dgl.to_csv('./RW_dgl_' + file_name + '_' + str(num_seeds) + '_.csv', mode='a', index=False, header=None)

            print(' ')

            
