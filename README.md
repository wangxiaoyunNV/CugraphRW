# CugraphRW
Originaly coding credit to Huang Xin, Xiaoyun made some changes

1 Donwload dataset

wget https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/test/datasets.tgz

2 extract the dataset

tar -xvzf datasets.tgz

3 rename the dataset

mv datasets data

4 To run the cugraph random walk, please use

python RW_cugraph_benchmark.py

5 you will get a bunch of .csv files.

