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


data = ['preferentialAttachment', 'as-Skitter', 'citationCiteseer', 'caidaRouterLevel', 'coAuthorsDBLP', 'coPapersDBLP']
#data = ['coAuthorsDBLP', 'coPapersDBLP']

for file_name in data:
    # cugraph RW
    t1 = time.time()
    G_cu = read_and_create('./data/'+ file_name + '.mtx')
    t2 = time.time() - t1
    print (t2)
    nodes = G_cu.nodes().to_array().tolist()
    num_nodes = G_cu.number_of_nodes()
    
    # some parameters
    num_seeds_ = [1000, 3000, 5000, 10000, 20000, 40000, 75000, 100000, 150000, 200000, 250000, 300000]
    max_depth_ = np.arange(2,2**2+1,2)
    for max_depth in max_depth_:
        for num_seeds in num_seeds_:
            if num_seeds >= num_nodes or ((file_name =='citationCiteseer' and num_seeds >200000) or (file_name =='coAuthorsDBLP' and num_seeds >= 150000)):
                break 
            
            print('number of seeds:', num_seeds)
            print('RW length:', max_depth)
            t_cugraph = []
            for i in range(11):
                seeds = random.sample(nodes, num_seeds)
                cu_seeds = cudf.Series(seeds)
                print (cu_seeds)
                t1 = time.time()
                subgraph = cugraph.subgraph(G_cu, cu_seeds)
                t2 = time.time() - t1
                print (t2)

            #df_t_cugraph = pd.DataFrame([t_cugraph])
            #df_t_cugraph.to_csv('./RW_cugraph_' + file_name + '_' + str(num_seeds) + '_.csv', mode='a', index=False, header=None)
    
