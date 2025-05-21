#!/bin/bash
#SBATCH --account=pi-aaz
#SBATCH --partition=gpu #the partition needs to be gpu for gpu jobs. Please review the user guide.
#SBATCH --output=op.out
#SBATCH --error=error.err # adding error output file
##SBATCH --array=0-21         # Commented out the line. The maximum array job is 12# run parallel in hardware level
##SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6 # requesting 6 CPUs to drive the GPU
#SBATCH --gres=gpu:1         # Request 1 GPU
##SBATCH --cpus-per-task=1 # commented out the line due to redundency
#SBATCH --constraint=rtx6000
##SBATCH --mem-per-gpu=80G # do you really need 80G? 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yukileong1@rcc.uchicago.edu

# Load necessary modules
module load python/anaconda-2022.05
echo "Loaded python/anaconda-2022.05"
#module list
# conda env list
source activate base
conda activate pytorch
# Run the Python script
echo "starting python script"
python3 finetune/ft_llama.py  # Replace 'your_script.py' with the script containing the zero-shot task code
echo "finished python script"

#python3: can't open file '/project/aaz/ashmitam/new_training.py': [Errno 2] No such file or directory
