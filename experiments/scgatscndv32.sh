#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=72G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=scgatscndv2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<ngravindra@gmail.com>
#SBATCH --output=slurm_%j.out


echo "starting %j with args $1,$2"
module restore cuda101
conda activate py38dev

mkdir -p ./logs

python -u /home/ngr4/project/scnd/scripts/experiments.py --exp "$1" --target "$2" --trial "$3" > ./logs/"$1"_"$2"_n"$3".log
