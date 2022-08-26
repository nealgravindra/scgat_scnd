#!/bin/bash

#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=108G
#SBATCH --time=1-0:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=<ngravindra@gmail.com>

conda activate py38dev
job_name=$(date +"%y%m%d.%H:%M")
mkdir -p ./logs

python -u /home/ngr4/project/scnd/scripts/pp_adata.py > ./logs/pp_adata_"$job_name".log 2>&1  