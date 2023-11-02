#!/bin/bash
         
#SBATCH --account=ctb-gwtaylor
#SBATCH --partition=c-gwtaylor
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:4
#SBATCH --cpus-per-task=4       
#SBATCH --mem=16000M            
#SBATCH --time=2-1:00:00
#SBATCH --output=%N-%j-GPU-Job.out    # %N for node name, %j for jobID
#SBATCH --exclusive

module load python/3.9 cuda cudnn

#Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

#Install all libraries

cd /home/pmillana/bioscan/paper/BarcodeBERT
pip install --no-index -r requirements.txt
echo 'Libraries Installed'

python MGPU_MLM_train.py --input_path=../../data/pre_training.tsv --k_mer=4 --stride=4 --checkpoint=True
