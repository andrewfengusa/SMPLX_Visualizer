# SMPLX_Visualizer

A simple offline video renderer to visualize the TED-SMPLX dataset.

## Setup Instructions
```bash
# Create python 3.7 virtual environment via conda
conda create -n smplx_viz python=3.8
conda activate smplx_viz
# Install required packages
pip smplx human_body_prior
# run example rendering script
python SMPLX_render.py
```