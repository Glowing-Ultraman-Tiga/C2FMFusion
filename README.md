C2FMFusion: A Multi-Modal and Multi-Scale BEV Feature Fusion Method for Autonomous Driving

This repository contains the official PyTorch implementation of C2FMFusion, a multi-modal BEV feature fusion framework for 3D object detection and BEV map segmentation in autonomous driving.

🔍 Overview

C2FMFusion addresses key challenges in camera–LiDAR BEV feature fusion, including:

Cross-modal feature misalignment

Limited receptive fields in existing fusion methods

Inefficient multi-scale feature interaction

The proposed framework introduces:

Uncertainty-aware Global Alignment Module (GAM) for adaptive modality weighting

Inter-Head Feature Interaction (CHI) to enhance semantic complementarity across attention heads

Coarse-to-Fine Multi-Scale Fusion (C2FF) with efficient attention for reduced computational cost

Extensive experiments on nuScenes and Waymo demonstrate consistent improvements in both detection accuracy and BEV map segmentation performance.

🧠 Method Overview
Camera BEV Features ─┐
                     ├─> Global Alignment + Uncertainty Branch ─┐
LiDAR BEV Features ──┘                                          │
                                                                ├─> Inter-Head Attention Interaction
Multi-Scale Features ──────────────────────────────────────────┘
                                   │
                          Coarse-to-Fine Fusion
                                   │
                    3D Detection & BEV Map Segmentation

🛠️ Installation
Requirements

Python ≥ 3.8

PyTorch ≥ 1.10

CUDA ≥ 11.1

mmcv / mmdet3d (recommended)

git clone https://github.com/yourname/C2FMFusion.git
cd C2FMFusion
pip install -r requirements.txt

📁 Dataset Preparation
nuScenes

Download the dataset from the official website:
https://www.nuscenes.org/

Follow the standard preprocessing steps used in BEVFusion / mmdet3d.

Update dataset paths in:

configs/_base_/datasets/nuscenes.py

Waymo

Follow the official Waymo Open Dataset instructions:
https://waymo.com/open/
