import pandas as pd
import os
import glob
from scipy import stats

work_path = 'a100-results'
write_path = 'a100_speedup/'

qry = work_path + '/*.csv'
files = glob.glob(qry)
#print (files)
methods = ['dgl','cugraph']
nseeds  = ['1000', '3000', '5000', '10000', '20000', '40000', '75000', '100000']
datasets = ['as-Skitter', 'caidaRouterLevel', 'citationCiteseer', 'coAuthorsDBLP', 'coPapersDBLP', 'preferentialAttachment']

result_fn = []


for data in datasets:
    result_name = 'gmean_'+ data + '_.csv'
    result_fn += [result_name]


for data in datasets:
    fn_speedup = write_path + data + '_speedup.csv'
    df_speedup = pd.DataFrame()
    for seed in nseeds:
        fn_cugraph = work_path + '/RW_cugraph_' + data + '_' + seed + '_.csv'
        fn_dgl = work_path + '/RW_dgl_' + data + '_' + seed + '_.csv'
        data_cugraph = pd.read_csv(fn_cugraph, header=None)
        data_dgl = pd.read_csv(fn_dgl, header=None)
        cugraph_gmean = stats.gmean(data_cugraph.iloc[:, 1:10], axis=1)
        dgl_gmean =  stats.gmean(data_dgl.iloc[:, 1:10], axis=1)
        speedup = dgl_gmean/cugraph_gmean
        df_speedup['cugraph_'+seed] = cugraph_gmean
        df_speedup['dgl_'+seed] = dgl_gmean
        df_speedup['speedup_'+seed] = speedup

    df_speedup.to_csv(fn_speedup,index=False)

