# **Standardized Fundus Glaucoma Dataset Classification**
This repository contains code for training Resnet50 and Xception classifiers for the Standardized Fundus Glaucoma Dataset (SMDG). Additionally, it also includes implementation of GradCAM (Gradient-weighted Class Activation Mapping) for visualizing the regions of the input image that are most important for the classification decision made by the models.

## Dataset
The Standardized Fundus Glaucoma Dataset (SMDG) is a publicly available dataset consisting of fundus images of healthy subjects and subjects with glaucoma. The dataset is contained in a archive folder consisting of an image folder and a metadata file in csv format. The images are in the png format and have a resolution of 512x512 pixels. The dataset can be downloaded from [here](https://drive.google.com/file/d/1hPu-xyzP88qWd-2lkZbnJHZ8RYMSep-q/view?usp=sharing)

## Dependencies
The following dependencies are required to run the code:

- Python 3.x
- Tensorflow
- Keras
- NumPy
- OpenCV
- Matplotlib

## Usage
1. Data
The dataset should be unzipped and placed in the main directory alongside train.ipynb within the repository. 

2. Train and evaluate models
- The train_resnet.ipynb and train_xception.ipynb file contains the code for preprocessing the dataset and training the ResNet50 and Xception models as well as the GRADCam visualization for both models.
- The model.py file contains the code for building the Resnet50 and Xception models. The eval_resnet.ipynb and eval_xception.ipynb notebook contains the code for evaluating the models.

3. Visualize GradCAM
The cam.py file contains the code for implementing GradCAM. Both training and validation notebooks contains GRADcam visualization for last layers of the models.

Training Metrics Wandb [here](https://wandb.ai/webcoders/Glaucoma-Detection-Backup?workspace=user-krakenb)
