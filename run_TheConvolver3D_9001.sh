#!/bin/sh
#BSUB -q c02516
#BSUB -J MULT
### number of core
#BSUB -n 4
### specify that all cores should be on the same host
#BSUB -gpu "num=1:mode=exclusive_process"
### specify the memory needed
#BSUB -R "rusage[mem=10GB]"
### Number of hours needed
#BSUB -W 12:00
### added outputs and errors to files
BSUB -o outputs/Output_%J.out
BSUB -e outputs/Error_%J.err

echo "Running script..."

module load cuda/11.8
module load python3/3.10.13
source /zhome/b3/8/148387/deep_learn_CV/.venv/bin/activate
python3 /zhome/b3/8/148387/deep_learn_CV/VideoClassification/SimpleTrainer.py > log/modular_train_test$(date +"%d-%m-%y")_$(date +'%H:%M:%S').log