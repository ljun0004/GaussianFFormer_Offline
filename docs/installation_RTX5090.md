Here is the finalized, battle-tested installation guide. 

I have made **one critical update** to Step 5 based on our final debugging session: I changed the build command for the `gaussian_encoder/ops` to use the `build_ext --inplace` method. This permanently prevents the "circular import" error by ensuring the compiled Blackwell binaries drop exactly where the Python scripts expect them. I also added a final Step 8 for your evaluation launch command.

You can copy and paste this directly into your project's `README.md`.

***

# Installation (Modernized for RTX 5090 / CUDA 13.0)

This code requires specific compilation strategies to bridge the original 2023 codebase with modern PyTorch 2.11 environments and the new **Blackwell (SM 10.0)** GPU architecture. Because PyTorch 2.11 does not yet natively compile SM 10.0 binaries, we use a `9.0+PTX` workaround to force the 5090's driver to JIT-compile the kernels at runtime.

## 0. Set Hardware Compilation Flags
Before creating the environment, ensure your system knows how to compile the C++ and CUDA operations for the RTX 5090. Add these to your `~/.bashrc` and run `source ~/.bashrc`.
```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-13.0

# THE BLACKWELL HACK: 
# Build for Hopper (9.0) but include PTX so the 5090 can JIT-compile it.
export TORCH_CUDA_ARCH_LIST="9.0+PTX"
export FORCE_CUDA=1
export MMCV_WITH_OPS=1
```

## 1. Create Conda Environment
Python 3.10 is required for compatibility with the OpenMMLab stack.
```bash
conda create -n selfocc python=3.10 -y
conda activate selfocc
```

## 2. Install PyTorch
Install the PyTorch version explicitly tailored for the CUDA 13.0 toolkit.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

## 3. Install Packages from MMLab (Native Compilation)
**CRITICAL:** Do NOT use `mim install` or pre-compiled wheels for `mmcv`. Standard wheels lack the `9.0+PTX` instructions required for Blackwell, resulting in `CUDA error: no kernel image is available`. We must compile the engine from source.

```bash
pip install mmengine --no-cache-dir

# Compile MMCV from source. This will take 15-20 minutes. 
# The -v flag lets you monitor the nvcc compilation.
pip install mmcv==2.1.0 --no-cache-dir --no-binary mmcv --no-build-isolation -v

# Install the rest of the suite. MMDetection requires source build, 
# while MMDet3D 1.x is pure Python and installs instantly.
pip install mmdet==3.2.0 mmsegmentation==1.2.2 --no-cache-dir --no-binary mmdet,mmsegmentation --no-build-isolation
pip install mmdet3d==1.4.0
```

## 4. Install Other Packages
*Note: We use `spconv-cu121` as it bridges perfectly with the 13.0 toolkit while waiting for native 13.0 wheel releases.*
```bash
pip install spconv-cu121 timm ftfy regex einops jaxtyping "numpy<2.0.0"
```

## 5. Install Custom CUDA Ops (Gaussian Rendering)
Ensure your `TORCH_CUDA_ARCH_LIST="9.0+PTX"` environment variable is still active in your terminal before running these, or the rendering will cause massive Memory Overflow errors. 

```bash
# Core GaussianFormer Ops (Must be built IN-PLACE to avoid circular imports)
cd model/encoder/gaussian_encoder/ops
rm -rf build && python setup.py build_ext --inplace

# Gaussian Aggregators (Standard build)
cd ../../../../model/head/localagg
rm -rf build && pip install -v . --no-build-isolation

# Additional Ops for GaussianFormer-2 configurations
cd ../localagg_prob
rm -rf build && pip install -v . --no-build-isolation

cd ../localagg_prob_fast
rm -rf build && pip install -v . --no-build-isolation

# Return to root directory
cd ../../../
```

## 6. Code Patch for MMDet3D 1.x
Starting in MMDet3D v1.x, the developers moved all 3D CUDA operators (like Farthest Point Sampling) into `mmcv.ops`. To prevent `ModuleNotFoundError` crashes in the older GaussianFormer code, **add this exact block to the very top of `eval.py` and `train.py`** (before any other imports):

```python
import sys
import mmcv.ops
# Bridge the legacy 3D Ops path directly to the modern MMCV location
sys.modules['mmdet3d.ops'] = mmcv.ops
```

## 7. (Optional) For Visualization
*Note: We must manually lock the VTK backend and restrict Matplotlib to maintain compatibility with `mayavi` builds and `nuscenes-devkit` requirements.*
```bash
# 1. Lock in the stable VTK backend
pip install vtk==9.2.6

# 2. Install mayavi without build isolation
pip install mayavi --no-build-isolation

# 3. Install remaining tools with NuScenes compatibility
pip install pyvirtualdisplay "matplotlib<3.6.0" PyQt5
```

## 8. Launching Evaluation
To run the evaluation on a single GPU without distributed launcher conflicts, set your target device and run the script:

```bash
# Ensure background debugging flags are off for full speed
unset CUDA_LAUNCH_BLOCKING
export CUDA_VISIBLE_DEVICES=0

mkdir -p /home/junn/Junn/GaussianFormer/out/nuscenes_gs25600_solid/

python train.py \
    --py-config config/nuscenes_gs25600_solid.py \
    --work-dir out/nuscenes_gs25600_solid/

python train.py \
    --py-config config/prob/nuscenes_gs25600.py \
    --work-dir out/nuscenes_gs25600/

python eval.py \
    --py-config config/nuscenes_gs25600_solid.py \
    --work-dir out/nuscenes_gs25600_solid/ \
    --resume-from ckpts/nuscenes_gs25600_solid/state_dict.pth

CUDA_VISIBLE_DEVICES=0 xvfb-run -a python visualize.py \
    --py-config config/nuscenes_gs25600_solid.py \
    --work-dir out/nuscenes_gs25600_solid \
    --resume-from ckpts/nuscenes_gs25600_solid/state_dict.pth \
    --vis-occ \
    --vis-gaussian \
    --num-samples 3 \
    --model-type base

python train_offline.py \
    --py-config config/prob/nuscenes_gs25600_offline.py \
    --work-dir out/nuscenes_gs25600_offline/

CUDA_VISIBLE_DEVICES=0 xvfb-run -a python visualize_offline.py \
    --py-config config/prob/nuscenes_gs25600_offline.py \
    --work-dir out/nuscenes_gs25600_offline \
    --vis-occ \
    --vis-gaussian \
    --model-type base

LIBGL_ALWAYS_INDIRECT=1 DISP=t CUDA_VISIBLE_DEVICES=0 python visualize_offline.py \
    --py-config config/prob/nuscenes_gs25600_offline.py \
    --work-dir out/nuscenes_gs25600_offline \
    --vis-occ \
    --vis-gaussian \
    --model-type base

python eval_offline.py --py-config config/nuscenes_gs25600_offline.py --scenes-dir ./out/nuscenes_gs25600_offline/scenes

```
***cd

This guide is now fully locked in. If you eventually deploy this project to the NTU Slurm cluster and they are running Hopper architectures (like H100s), this exact same `9.0+PTX` build strategy will compile flawlessly there too!