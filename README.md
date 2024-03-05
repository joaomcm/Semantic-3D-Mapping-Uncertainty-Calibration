# GLFS
This repository contains the code for reproducing the work of the ICRA 2024 paper "On the Overconfidence Problem in Semantic 3D Mapping", as well as the code for tuning the parameters of the Generalized Learned Fusion Strategy (GLFS) introduced in this paper.


![scene0699_00](https://github.com/joaomcm/Semantic-3D-Mapping-Uncertainty-Calibration/assets/27590978/17eadbbc-0915-493e-ba56-92144e73442e)

This repository serves 3 purposes: 

1) To provide a simple python wrapper for generalized semantic 3D reconstruction on the GPU

2) To enable the reproduction of our experimental results on calibration

3) To provide template code for training the parameters of your own GLFS model for your application

The instructions for each is provided in its appropriate section of this README.

## Requirements

Our reconstruction code should be able to run on any cuda-enabled gpu with at least 8 Gb of VRAM. Some dataset generation parts of the script might be more memory intensive, demanding upwards of 30Gb of RAM. This, however, can be adjusted by adjusting chunking parameters on the dataset creation scripts. For any questions on how to do that, please submit an issue to this repo. 

All of our experiments are able to run on a machine with a i7-9700K processor, 64Gb of RAM and an RTX 3090 GPU. 

## To use our simple wrapper for different reconstruction pipelines

For the first use case, you can simply copy reconstruction.py into your project and see the top level example.py for reference on how to use the different kinds of reconstruction pipelines described in our paper.

## Setup instructions for reproducing paper results

1) Clone this repository and its submodules : ``` git clone --recurse-submodules git@github.com:joaomcm/Semantic-3D-Mapping-Uncertainty-Calibration.git ```.

2) Create the virtual environment, e.g. ``` conda env create -n learned_3d_fusion python=3.7 ``` .

3) Activate the virtual environment: ``` conda activate learned_3d_fusion ```.

4) Install almost all the required packages: ``` pip install -r requirements.txt ```.

5) Install torch for your cuda version as in : ```  pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1  --extra-index-url https://download.pytorch.org/whl/cu117 ```.

6) Download the pretrained weights for the finetuned Segformer (https://uofi.box.com/s/lnuxvqh77tulivbew7c9y0m6jh5y23ti) and ESANet (https://uofi.box.com/s/hd3mlqcnwh9k1i3f5ffur5kcup32htby) models and place them in their respective /segmentation_model_checkpoints folder.

7) Fill in the directories you wish to use for the data, results, etc at settings/directory_definitions.json.

8) Obtain the ScanNet v2 dataset (https://github.com/ScanNet/ScanNet)

9) Create the ground truth reconstruction files by executing ```python create_reconstruction_gts.py```

10) Run ```python perform_reconstruction.py``` to run all the reconstruction experiments described in experiments_and_short_names

11) Once all reconstructions are done, run ```python run_full_eval.py```, which will output the summarized results to the Results/quant_eval folder.


For reproducing our ObjectGoalNavigation results, please check our fork of PEANUT (https://github.com/joaomcm/PEANUT/tree/traditional_mapping)- since the code for those experiments has been kept on a separate repository for organization reasons. 


