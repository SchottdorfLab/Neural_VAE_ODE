#!/bin/bash -l
#SBATCH --job-name=v6_time
#SBATCH --partition=gpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/v6-time-%j.out
#SBATCH --error=logs/v6-time-%j.err

cd ~/projects/Neural_VAE_ODE

vpkg_devrequire miniconda/25.1.1.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/envs/odevae

export MPLBACKEND=Agg

which python
python --version
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

python -u src/v6_binned_raw_eval_ae.py --config configs/v6_binned_raw_eval_ae_time.txt
