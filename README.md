# ENTENTE: Cross-silo Intrusion Detection on Network Log Graphs with Federated Learning
Our paper `ENTENTE: Cross-silo Intrusion Detection on Network Log Graphs with Federated Learning` will be presented at the NDSS Symposium 2026!

## Citation
```
@article{xu2025entente,
  title={Entente: Cross-silo Intrusion Detection on Network Log Graphs with Federated Learning},
  author={Xu, Jiacen and Li, Chenang and Zheng, Yu and Li, Zhou},
  journal={arXiv preprint arXiv:2503.14284},
  year={2025}
}
```

## Setup

#### Python Environment
Create enviroment and install required packages

For GPU enviroment:
```bash
conda create -n entente python==3.9 -y
conda activate entente
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.10.1+cu111.html --no-index
```
For CPU only enviroment:
```bash
conda create -n entente python==3.9 -y
conda activate entente
pip install torch==1.10.1 torchvision==0.11.2
pip install -r requirements.txt
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.10.1+cpu.html --no-index
```
#### Datasets
Download the pre-processed [datasets](https://zenodo.org/records/16014433). Put the `data` folder at the root of this repo.

## Euler
```
    python Euler/run.py --cluster_fname optc_2_2.json --client_number 3 --epochs 1
    python Euler/run.py --cluster_fname optc_3_3.json --client_number 4 --epochs 1
```

## Jbeil
```
    python Jbeil/run.py --client_number 3
    python Jbeil/run.py --client_number 4
```
