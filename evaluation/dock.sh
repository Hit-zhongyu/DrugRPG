#! /bin/bash
#SBATCH --job-name=evaluate3
#SBATCH --time=48:00:00
#SBATCH -o /mnt/rna01/lzy/pocketdiff7/evaluation/slurm3.out
#SBATCH -p Normal -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=150G


source /mnt/rna01/lzy/.bashrc
conda activate pocketdiff2

# python -u /mnt/rna01/lzy/Pocketdiff/evaluate.py --path /mnt/rna01/lzy/pocketdiff7/logging/sample_weight3_guidance3_6_85pt/samples_all.pkl > /mnt/rna01/lzy/pocketdiff7/valid_weight3_guidance3_6_85pt_731.txt


# python -u /mnt/rna01/lzy/Pocketdiff/evaluate.py --path /mnt/rna01/lzy/pocketdiff5/logging/4aua/samples_all.pkl > /mnt/rna01/lzy/pocketdiff5/valid_4aua.txt
# for single
python -u /mnt/rna01/lzy/Pocketdiff/evaluate_single.py --path /mnt/rna01/lzy/targetdiff/outputs_pdb_7ran/samples_all.pkl  --pdb_path /mnt/rna01/lzy/pocketdiff7/7RAN/7ran_pocket10.pdb  > /mnt/rna01/lzy/targetdiff/outputs_pdb_7ran/valid_7ran_5k.txt 