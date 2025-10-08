#!/bin/bash
#$ -cwd                 
#$ -pe smp 6
#$ -l h_rt=1:0:0
#$ -l h_vmem=32G
#$ -l highmem
#$ -j n
#$ -o HPC_Files/logo/
#$ -e HPC_Files/loge/

set -e 

# --- Environment Setup ---
module load python
source /data/home/qc25022/cancer-extraction-pipeline/env/bin/activate

# python -u src/pipeline/debug_patient_trajectory.py
python -u main.py --stage 3
#python -u src/utils/analyse_mappings.py




