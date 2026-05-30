#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ==========================================
# Setup Directories
# ==========================================
# Define base directory
BASE_DIR="/projects/faculty/tay_wee_peng_2026_05/GaussianFFormer/data"

echo "=========================================="
echo "Starting data preparation at $(date)"
echo "=========================================="

echo "Creating directory structure in $BASE_DIR..."
mkdir -p $BASE_DIR/nuscenes
mkdir -p $BASE_DIR/nuscenes_cam
mkdir -p $BASE_DIR/surroundocc/samples

# ==========================================
# 1. nuScenes V1.0 Dataset
# ==========================================
echo "Processing nuScenes..."
cd $BASE_DIR/nuscenes

# Check using the automatically extracted metadata txt file
if [ ! -f ".v1.0-trainval_meta.txt" ]; then
    echo "Downloading Train/Val Metadata..."
    wget -c "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz"
    tar -xzvf v1.0-trainval_meta.tgz
    rm v1.0-trainval_meta.tgz
else
    echo "Train/Val Metadata already extracted. Skipping."
fi

for i in {01..10}; do
    # Check using the automatically extracted blob txt file
    if [ ! -f ".v1.0-trainval${i}_blobs.txt" ]; then
        echo "Downloading and extracting Train/Val Blob Part ${i}..."
        wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval${i}_blobs.tgz"
        tar -xzvf v1.0-trainval${i}_blobs.tgz
        rm v1.0-trainval${i}_blobs.tgz
    else
        echo "Train/Val Blob Part ${i} already extracted. Skipping."
    fi
done

if [ ! -f ".v1.0-test_meta.txt" ]; then
    echo "Downloading Test Metadata..."
    wget -c "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-test_meta.tgz"
    tar -xzvf v1.0-test_meta.tgz
    rm v1.0-test_meta.tgz
else
    echo "Test Metadata already extracted. Skipping."
fi

if [ ! -f ".v1.0-test_blobs.txt" ]; then
    echo "Downloading Test Blobs..."
    wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-test_blobs.tgz"
    tar -xzvf v1.0-test_blobs.tgz
    rm v1.0-test_blobs.tgz
else
    echo "Test Blobs already extracted. Skipping."
fi

# ==========================================
# 2. SurroundOcc Annotations
# ==========================================
echo "Processing SurroundOcc..."
cd $BASE_DIR/surroundocc

if [ -f "$BASE_DIR/surroundocc.zip" ]; then
    echo "Extracting main surroundocc.zip via Python..."
    python3 -c "
import zipfile
with zipfile.ZipFile('$BASE_DIR/surroundocc.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
"

    # Extract the nested zips
    if [ -f "train.zip" ]; then
        echo "Extracting train.zip via Python..."
        python3 -c "
import zipfile
with zipfile.ZipFile('train.zip', 'r') as zip_ref:
    zip_ref.extractall('samples/')
"
        rm train.zip
    fi

    if [ -f "val.zip" ]; then
        echo "Extracting val.zip via Python..."
        python3 -c "
import zipfile
with zipfile.ZipFile('val.zip', 'r') as zip_ref:
    zip_ref.extractall('samples/')
"
        rm val.zip
    fi
else
    echo "Warning: $BASE_DIR/surroundocc.zip not found. Please upload it first."
fi

# ==========================================
# 3. nuScenes-Cam PKL Files (Local Zip Method)
# ==========================================
echo "Processing PKL files..."
cd $BASE_DIR/nuscenes_cam

# Check if any of the final pkl files are missing before extracting
if [ ! -f "nuscenes_infos_train_sweeps_occ.pkl" ] || [ ! -f "nuscenes_infos_val_sweeps_occ.pkl" ] || [ ! -f "nuscenes_infos_val_sweeps_lid.pkl" ]; then
    if [ -f "$BASE_DIR/nuscenes_infos.zip" ]; then
        echo "Extracting local nuscenes_infos.zip via Python..."
        python3 -c "
import zipfile
with zipfile.ZipFile('$BASE_DIR/nuscenes_infos.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
"
    else
        echo "Warning: $BASE_DIR/nuscenes_infos.zip not found at the base data root path."
    fi
else
    echo "All target PKL information files are already extracted. Skipping."
fi

echo "=========================================="
echo "Data preparation script completed at $(date)"
echo "=========================================="