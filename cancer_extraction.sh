#!/bin/bash
#$ -cwd                 
#$ -pe smp 12
#$ -l h_rt=1:0:0
#$ -l h_vmem=4G
#$ -j n
#$ -o HPC_Files/logo/
#$ -e HPC_Files/loge/

set -e 

# --- Environment Setup ---
module load python
source /data/home/qc25022/cancer-extraction-pipeline/env/bin/activate

# python -u src/pipeline/step_01_define_cohort.py
python -u main.py --stage 5
#python -u src/utils/analyse_mappings.py




