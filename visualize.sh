#!/bin/sh
#SBATCH -p et2
#SBATCH --mem=30000
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --output=vis%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bs407@duke.edu

pwd; hostname; date

python3.7 visualize.py

date


