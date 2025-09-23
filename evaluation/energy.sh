#! /bin/bash
#SBATCH --job-name=energy_PMDM3
#SBATCH --time=72:00:00
#SBATCH -o /mnt/rna01/lzy/pocketdiff7/evaluation/energy_PMDM3.out
#SBATCH -p Normal -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=100G


source /mnt/rna01/lzy/.bashrc
conda activate pocketdiff2

# python -u /mnt/rna01/lzy/Pocketdiff/evaluate.py --path /mnt/rna01/lzy/pocketdiff7/logging/sample_weight1_guidance3_6_90pt/samples_all.pkl > /mnt/rna01/lzy/pocketdiff7/valid_weight1_guidance3_6_90pt_724.txt


# python -u /mnt/rna01/lzy/Pocketdiff/evaluate.py --path /mnt/rna01/lzy/pocketdiff5/logging/4aua/samples_all.pkl > /mnt/rna01/lzy/pocketdiff5/valid_4aua.txt
# for single
# python -u /mnt/rna01/lzy/Pocketdiff/evaluate_single.py --path /mnt/rna01/lzy/pocketdiff7/custom_pdb/sample_7ran_8k_guidance36_120pt_new/samples_all.pkl --pdb_path /mnt/rna01/lzy/pocketdiff7/7RAN/7ran_pocket10.pdb  > /mnt/rna01/lzy/pocketdiff7/valid_7ran_8k_guidance36_120pt.txt 

# python -u /mnt/rna01/lzy/Pocketdiff/evaluatte_energy3.py  --path /mnt/rna01/lzy/pocketdiff7/logging/sample_weight1_guidance3_6_88pt/samples_all.pkl >  /mnt/rna01/lzy/pocketdiff7/energy_valid_88_new3.txt
# python -u /mnt/rna01/lzy/Pocketdiff/evaluate_energy_UFF.py  --path /mnt/rna01/lzy/pocketdiff7/logging/sample_weight1_guidance3_6_88pt/samples_all.pkl >  /mnt/rna01/lzy/pocketdiff7/energy_valid_88_new.txt

# python -u /mnt/rna01/lzy/Pocketdiff/evaluatte_energy3.py  --path /mnt/rna01/lzy/compare/sample_targetdiff_10000.pkl > /mnt/rna01/lzy/compare/energy_targetdiff3.txt
# python -u /mnt/rna01/lzy/Pocketdiff/evaluate_energy_UFF.py  --path /mnt/rna01/lzy/compare/sample_targetdiff_10000.pkl > /mnt/rna01/lzy/compare/energy_targetdiff_new.txt

# python -u /mnt/rna01/lzy/Pocketdiff/evaluatte_energy3.py  --path /mnt/rna01/lzy/compare/samples_diffsbdd_10000_202496.pkl > /mnt/rna01/lzy/compare/energy_diffsbdd3.txt
#  python -u /mnt/rna01/lzy/Pocketdiff/evaluate_energy_UFF.py  --path /mnt/rna01/lzy/compare/samples_diffsbdd_10000_202496.pkl > /mnt/rna01/lzy/compare/energy_diffsbdd_new.txt

# python -u /mnt/rna01/lzy/Pocketdiff/energy_decom3.py --path /mnt/rna01/lzy/compare/sample_decomp_9800.pkl > /mnt/rna01/lzy/compare/energy_decomp3.txt
# python -u /mnt/rna01/lzy/Pocketdiff/energy_decom_uff.py --path /mnt/rna01/lzy/compare/sample_decomp_9800.pkl > /mnt/rna01/lzy/compare/energy_decomp_new.txt

python -u /mnt/rna01/lzy/Pocketdiff/energy_pmdm3.py --path /mnt/rna01/lzy/PMDM/generalizedbuild_0_100_result_2025_03_25__21_46_44/samples_all.pkl > /mnt/rna01/lzy/compare/energy_PMDM3.txt
# python -u /mnt/rna01/lzy/Pocketdiff/energy_pmdm_uff.py --path /mnt/rna01/lzy/PMDM/generalizedbuild_0_100_result_2025_03_25__21_46_44/samples_all.pkl > /mnt/rna01/lzy/compare/energy_PMDM_new.txt

# python -u /mnt/rna01/lzy/Pocketdiff/evaluatte_energy3.py  --path /mnt/rna01/lzy/compare/samples_all_4_pocket2mol.pkl  > /mnt/rna01/lzy/compare/energy_pocket2mol3.txt
# python -u /mnt/rna01/lzy/Pocketdiff/evaluate_energy_UFF.py  --path /mnt/rna01/lzy/compare/samples_all_4_pocket2mol.pkl  > /mnt/rna01/lzy/compare/energy_pocket2mol_new.txt

# python -u /mnt/rna01/lzy/Pocketdiff/evaluatte_energy3.py  --path /mnt/rna01/lzy/compare/samples_all_4_pocketflow.pkl > /mnt/rna01/lzy/compare/energy_pocketflow3.txt
# python -u /mnt/rna01/lzy/Pocketdiff/evaluate_energy_UFF.py  --path /mnt/rna01/lzy/compare/samples_all_4_pocketflow.pkl > /mnt/rna01/lzy/compare/energy_pocketflow_new.txt





