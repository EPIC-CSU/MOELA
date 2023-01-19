#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=amilan
#SBATCH --qos=long
#SBATCH --account=csu-general
#SBATCH --nodes=1
#SBATCH --ntasks=5                     # Specify Summit haswell nodes
#SBATCH --job-name=MOOS-5obj                   # Job submission name
#SBATCH --output=MOOS-5obj.%A-%a.out            # Output file name with Job ID
#SBATCH --array=1-8

module purge

module load anaconda

conda activate sirui-env

cd /scratch/alpine/qsr1024@colostate.edu/MOELA-5obj/

python random_seed_MOOS.py $(sed -n "${SLURM_ARRAY_TASK_ID}p" HPC_input_args)

conda deactivate
