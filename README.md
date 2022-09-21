# SMPLX_Visualizer

A simple offline video renderer to visualize the TED-SMPLX dataset.

## Setup Instructions

Download SMPL-X models from [here](https://smpl-x.is.tue.mpg.de/download.php) and put the model files (.npz) to `smplx_model` folder.

```bash
# Create python 3.8 virtual environment via conda
conda create -n smplx_viz python=3.8
conda activate smplx_viz
# Install required packages
pip smplx human-body-prior

# run example rendering script (mesh)
python SMPLX_mesh_render.py
# run example rendering script (skeleton)
python SMPLX_skeleton_render.py
```