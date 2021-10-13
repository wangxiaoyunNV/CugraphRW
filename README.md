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

6 to run DGL sampling code, you need to install pytorch and DGL first inside of the container. 

pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install dgl-cu111 -f https://data.dgl.ai/wheels/repo.html

7 run DGL random walk, it is similar to step 4.

python RW_DGL_benchmark.py
