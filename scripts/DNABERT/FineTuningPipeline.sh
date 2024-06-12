#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=0-23:00:00

module load python/3.9 cuda cudnn

#Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

#Install all libraries

cd /home/pmillana/bioscan/
pip install --no-index -r requirements.txt
echo 'Libraries Installed'
cd scripts/DNABERT/

#python supervised_learning.py --input_path=../../data -k 4 --model dnabert --checkpoint dnabert/4-new-12w-0

#python supervised_learning.py --input_path=../../data -k 6 --model dnabert --checkpoint dnabert/6-new-12w-0

python supervised_learning.py --input_path=../../data -k 5 --model dnabert --checkpoint dnabert/5-new-12w-0
