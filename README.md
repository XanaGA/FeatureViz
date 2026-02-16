# FeatureViz 🔬

**FeatureViz** is a modular library designed for the deep analysis and visualization of latent features from State-of-the-Art (SOTA) computer vision models (e.g., DinoV3, V-JEPA2, VGGT). 

The goal is to provide researchers with a standardized toolkit to explore how these models "see" the world, focusing on geometric consistency (rotations), statistical distributions, and feature correlations.

## 🚀 Quick Start

### 1. Environment Setup (Conda + Pip)
We recommend using Conda to manage the base environment and Pip for internal library development.

```bash
# Install FeatureViz in editable mode
git clone [https://github.com/XanaGA/featureviz.git](https://github.com/XanaGA/featureviz.git)
cd featureviz

# Create and activate environment
conda create -n featureviz_env python=3.12 -y
conda activate featureviz_env

# Install PyTorch (adjust cuda version as needed)
pip install torch==2.6.0 torchvision==0.21.0 --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
pip install -e .
```

