#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=2-1:00:00
#SBATCH --output=%N-%j-GPU-Job.out    # %N for node name, %j for jobID

module load python/3.11 cuda cudnn
ENVPATH="$SLURM_TMPDIR/env"
virtualenv --no-download "$ENVPATH"
source "$ENVPATH/bin/activate"
python -m pip install --no-index -r CC_requirements.txt
echo 'Libraries Installed'
pip install --no-index .
echo 'Installed'

python barcodebert/pretraining.py --run-name barcodebert_computecanada --k-mer 4 --stride 4 --data-dir ./data/
