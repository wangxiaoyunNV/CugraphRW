import pandas as pd
import os
import glob

work_path = 'results'
write_path = 'results_processed/'

qry = work_path + '/*.csv'
files = glob.glob(qry)

for fn in files:
    fnnames = os.path.splitext(os.path.split(fn)[1])[0]
    data = pd.read_csv(fn, header=None)
    data['mean'] = data.iloc[:, 1:10].mean(axis=1)
    data.to_csv(write_path + fnnames + 'processed.csv',index=False)
