<img src="imgs/model.jpg" alt="model" style="zoom: 40%;" />

# Rethinking Graph Masked Autoencoders through Alignment and Uniformity

Implementation for paper: Rethinking Graph Masked Autoencoders through Alignment and Uniformity

## Dependencies

* Python >= 3.7
* PyTorch >= 1.9.0 
* dgl >= 0.7.2
* pyyaml == 5.4.1

## Quick Start

For quick start, you could run the scripts:

**Node classification**

```bash
sh scripts/run_transductive.sh <dataset_name> <gpu_id> # for transductive node classification
# example: sh scripts/run_transductive.sh cora/citeseer/pubmed/ogbn-arxiv 0
sh scripts/run_inductive.sh <dataset_name> <gpu_id> # for inductive node classification
# example: sh scripts/run_inductive.sh reddit/ppi 0

# Or you could run the code manually:
# for transductive node classification
python main_transductive.py --dataset cora --seed 0 --device 0 --use_cfg
# for inductive node classification
python main_inductive.py --dataset ppi --seed 0 --device 0 --use_cfg
```
Supported datasets:
* transductive node classification:  `cora`, `citeseer`, `pubmed`,` corafull`, `wikics`,`ogbn-arxiv`,`flickr`
* inductive node classification: `ppi`, `reddit` 


**Graph classification**

```bash
sh scripts/run_graph.sh <dataset_name> <gpu_id>
# example: sh scripts/run_graph.sh mutag/imdb-b/imdb-m/proteins/... 0 

# Or you could run the code manually:
python main_graph.py --dataset IMDB-BINARY  --seed 0 --device 0 --use_cfg
```
Supported datasets: 

- `IMDB-BINARY`, `IMDB-MULTI`, `PROTEINS`, `MUTAG`,  `COLLAB`,`PTC-MR`,`REDDIT-BINERY`
