# GLFS
This repository contains the code for reproducing the work of the ICRA 2024 paper "On the Overconfidence Problem in Semantic 3D Mapping", as well as the code for tuning the parameters of the Generalized Learned Fusion Strategy (GLFS) introduced in this paper.


## Setup instructions

1) Create the virtual environment, e.g. ``` conda env create -n learned_3d_fusion python=3.7 ```

2) Activate the virtual environment: ``` conda activate learned_3d_fusion ```

3) Install almost all the required packages: ``` pip install -r requirements.txt ```

4) Install torch for your cuda version as in : ```  pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1  --extra-index-url https://download.pytorch.org/whl/cu117 ```