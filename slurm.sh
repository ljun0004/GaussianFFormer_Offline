#!/bin/bash

#SBATCH --job-name=GaussianFFormer_Offline
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/job-%j.out
##SBATCH --error=logs/job-%j.err

##SBATCH --gpus=pro6000:8 -C highmem
#SBATCH --gpus=pro6000:8

# REQUIRED: Project accounting and priority
#SBATCH -A faculty-proj
#SBATCH --qos=tay_wee_peng_2026_05

# NOTE: CPU and RAM are automatically scaled by Slurm (16 cores / 96 GiB RAM).
# Do not add --mem or --cpus-per-task.

# 1. Load the cluster environment
module load Miniforge3
module load CUDA/13.0.0
source activate selfocc

# 2. Set the "Fat Binary" Flags (Ampere + Ada Lovelace + Blackwell)
export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX"
export FORCE_CUDA=1
export MMCV_WITH_OPS=1

# 3. Ensure output directories exist
mkdir -p logs
mkdir -p out/nuscenes_gs25600_offline/

# 4. Auto-Compile Dependencies (Only runs once!)
echo "========================================"
if [ ! -f ".compiled_fat_binary" ]; then
    echo "First run detected: Compiling MMCV and ALL Gaussian Ops for all architectures..."
    
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

    # 1. Recompile MMCV
    # pip uninstall mmcv -y
    pip install mmcv==2.1.0 --no-cache-dir --no-binary mmcv --no-build-isolation -v

    # 2. Install the rest of the MMLab suite and dependencies
    pip install mmengine mmdet==3.2.0 mmsegmentation==1.2.2 mmdet3d==1.4.0 \
        --no-cache-dir \
        --no-binary mmdet,mmsegmentation \
        --no-build-isolation

    pip install spconv-cu121 timm ftfy regex einops jaxtyping "numpy<2.0.0"
    
    # 3. Recompile Gaussian Encoder Ops and Aggregators
    cd model/encoder/gaussian_encoder/ops
    rm -rf build && python setup.py build_ext --inplace
    
    cd ../../../../model/head/localagg
    rm -rf build && pip install -v . --no-build-isolation
    
    cd ../localagg_prob
    rm -rf build && pip install -v . --no-build-isolation
    
    cd ../localagg_prob_fast
    rm -rf build && pip install -v . --no-build-isolation
    
    # Return to root directory
    cd ../../../
    
    # Create the flag file so this block never runs again
    touch .compiled_fat_binary
    echo "Compilation complete! Flag file '.compiled_fat_binary' created."
else
    echo "Fat Binary already compiled (found .compiled_fat_binary). Skipping compilation!"
fi
echo "========================================"

# 5. Auto-Detect GPU Count via Hardware
export NPROC_PER_NODE=$(nvidia-smi -L | grep -c "GPU")
echo " Node: $(hostname)" 
echo " Auto-detected GPUs: ${NPROC_PER_NODE}" 
nvidia-smi -L
echo "========================================"

# Fail-Fast just in case nvidia-smi crashes or returns 0
if [ "$NPROC_PER_NODE" -eq 0 ]; then
    echo "FATAL ERROR: nvidia-smi detected 0 GPUs. Aborting."
    exit 1
fi

# 6. Launch the job using modern PyTorch DDP
echo "Starting distributed extraction on $NPROC_PER_NODE GPUs..."

torchrun --nproc_per_node=$NPROC_PER_NODE train_offline.py \
    --py-config config/prob/nuscenes_gs25600_offline.py \
    --work-dir out/nuscenes_gs25600_offline/

```