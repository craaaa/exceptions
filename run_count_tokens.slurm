#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=tokens
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --time=30:00
#SBATCH --mem=32G

singularity exec --overlay /scratch/cl5625/overlay-exceptions.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "

source /ext3/env.sh
conda activate exceptions

python $SCRATCH/exceptions/scripts/count_tokens.py --dataset_name $DATASET_NAME
"