# DrugRPG：Integrating Chemical Priors and Physical Laws with Diffusion Models for Structure-Based Drug Design
**Official implementation of DrugRPG, a structure-based drug design with representationo alignment and physics guidance, by Zhongyu Liu.<br>**

![image](https://github.com/Hit-zhongyu/DrugRPG/blob/main/image/DrugRPG.png)

# Dependencies
### **Conda environment**

Please use our environment file to install the environment.
```
conda create -n DrugRPG python=3.9
conda activate DrugRPG
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install pyg -c pyg
conda install rdkit openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge

# For Vina Docking
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
``` 
  
### **Pre-trained models**
The pre-trained models could be downloaded from [Zenodo](https://zenodo.org/records/17183753).

# Benchmarks

Download and extract the dataset is provided in [Zenodo](https://zenodo.org/records/17183753)

The original CrossDocked dataset can be found at https://bits.csb.pitt.edu/files/crossdock2020/

# Training 
```
TORCH_DISTRIBUTED_DEBUG=INFO OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0  torchrun --rdzv_backend=static --rdzv_endpoint=localhost:0 --standalone --nproc_per_node=1 train.py
```

# Inference
Sample molecules for all pockets in the test set
```
python sample_batch.py --ckpt <checkpoint> --num_atom <number of samples> --batch_size <batch_size> --cuda <cuda> 
```

Sample molecules for given customized pockets
```
python sample_for_pocket.py --pdb_path <pdb path>  --num_atom <number of samples> --batch_size <batch_size>   --ckpt <checkpoint>
```

# Evaluation
```
python -u evaluate.py --path <generate_molecule_path>
```
