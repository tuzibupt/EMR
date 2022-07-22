# EMR-GNN

This is the PyTorch implementation for our IJCAI 2022 paper:

Yuling Wang, Hao Xu, Yanhua Yu, Mengdi Zhang, Zhenhao Li, Yuji Yang, Wei Wu. 2022. Ensemble Multi-Relational Graph Neural Networks, [Paper in arXiv](https://arxiv.org/abs/2205.12076).

## Environment Requirement

- Conda >= 4.8
- Python >= 3.7
- torch>=1.8.1
- dgl >= 0.6.1

## About Data Format

Each line is split by '\t'. For each dataset, we have:

- **node.dat:**The information of nodes. Each line has (node_id, node_name, node_type_id, node_feature). One-hot node_features can be omitted. Each node type takes a continuous range of node_ids. And the order of node_ids are sorted by node_type_id, which means that node_type_id=0 takes the first interval of node_id, node_type_id=1 takes the second interval of node_id, etc. Node features are vectors split by comma.
- **link.dat:** The information of edges. Each line has (node_id_source, node_id_target, edge_type_id, edge_weight).
- **label.dat:** The information of node labels. Each line has (node_id, node_type_id, node_label). For multi-label setting, node_labels are split by comma.
- **label.dat.test:** Test set node labels. The format is same as label.dat, but the node_label is randomly replaced.

## Quick Start

**Firstly**, install all dependencies with the following command line:

`pip install -r requirements.txt`

**Secondly**, run entity_classify.py in IDE or with command line:

### MUTAG

`python entity_classify.py --dataset mutag --lamda1=100 --lamda2=10  --n-hidden=64 --hm_dim=64 --transf_type=1 -e=80 --seed=10`

### BGS

`python entity_classify.py --dataset bgs --lamda1=0.3 --lamda2=0.3  --n-hidden=20 --hm_dim=20 --transf_type=0 -e=25 --seed=10`

### ACM

`python entity_classify.py --dataset acm --lamda1=27 --lamda2=34  --feats-type=2 --seed=10`

### DBLP

`python entity_classify.py --dataset dblp  --lamda1=0.15 --lamda2=20  --feats-type=2 --seed=10`

## Note

The implemention is based on [HGB-RGCN](https://github.com/THUDM/HGB/tree/master/NC/RGCN), [Torch-RGCN](https://github.com/thiviyanT/torch-rgcn).