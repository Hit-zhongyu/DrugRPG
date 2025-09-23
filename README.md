# DrugRPG：Integrating Chemical Priors and Physical Laws with Diffusion Models for Structure-Based Drug Design
Official implementation of DrugRPG, a structure-based drug design with representationo alignment and physics guidance, by Zhongyu Liu.

# Dependencies
**Conda environment**

Please use our environment file to install the environment.
```
# Clone the environment
conda env create -f DrugRPG.yml
# Activate the environment
conda activate DrugRPG
```

Pre-trained models
The pre-trained models could be downloaded from Zenodo.

# Benchmarks

Download and extract the dataset is provided in Zenodo

The original CrossDocked dataset can be found at https://bits.csb.pitt.edu/files/crossdock2020/

# Training 
```
TORCH_DISTRIBUTED_DEBUG=INFO OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0  torchrun --rdzv_backend=static --rdzv_endpoint=localhost:0 --standalone --nproc_per_node=1 train.py
```
