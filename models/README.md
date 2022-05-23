All the models are in bert_hgt.py.

Including 
```
* BERT_HGT_CON: Build constituency graph on top of backbone
* BERT_HGT_DEP: Build dependency graph on top of backbone
* BERT_HGT_CON_AND_DEP: Build constituency and dependency graph on top of backbone
```
You you change the current backbone BERT to other tansformer-based models.

Currently, this work leverages [Heteorgeneous Graph Transformer (HGT)](https://arxiv.org/abs/2003.01332) to handle the graphical structures.
HGT was implemented with the PyTorch Geometric library. It can also be changed to other variables of graph neural networks.
More information please see the introductions here https://pytorch-geometric.readthedocs.io/en/latest/.
