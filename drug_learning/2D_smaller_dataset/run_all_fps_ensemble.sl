#!/bin/bash
#SBATCH --job-name="all_fps_ensemble"
#SBATCH -D .
#SBATCH --output=all_fps_ensemble.out
#SBATCH --error=all_fps_ensemble.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
##SBATCH --qos=debug



export PYTHONPATH="/home/bsc72/bsc72021/drug_lerning_repos/drug_learning/drug_learning"
source activate drug_learning

# python model_evaluation_SARS2_multicoupling.py yaml_files/all_fps_SVM_SARS2.yaml
# python model_evaluation_SARS2_multicoupling.py yaml_files/all_fps_RF_SARS2.yaml
# python model_evaluation_SARS2_multicoupling.py yaml_files/all_fps_SVM_opt_param_SARS2.yaml
python model_evaluation_SARS2_multicoupling.py  yaml_files/all_fps_knn_SARS2.yaml
python model_evaluation_SARS2_multicoupling.py  yaml_files/all_fps_voting_SARS2.yaml
