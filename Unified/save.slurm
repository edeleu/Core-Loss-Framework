#!/bin/bash
#SBATCH --job-name=SaveTransformer    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1
#SBATCH --time=2:23:03          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=ed5754@princeton.edu

module purge
module load anaconda3/2023.3
conda activate /scratch/gpfs/sw0123/torch-compile
cd /scratch/gpfs/sw0123/

python -u UnifiedMultiMaterialHack.py