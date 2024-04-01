#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=count
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

singularity exec --nv --overlay /scratch/cl5625/overlay-exceptions.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "

source /ext3/env.sh
conda activate corpus

echo $SOURCE_V REDUCE $TARGET_V
python $SCRATCH/exceptions/scripts/reduce_counts.py -s $SOURCE_V -t $TARGET_V
"