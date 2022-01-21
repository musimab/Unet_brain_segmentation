## U-Net for Brain segmentation

Loading model using PyTorch Hub: pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet

import torch
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)

## Input data

Dataset used for development and evaluation was made publicly available on Kaggle


## torchscript

TorchScript is a way to create serializable and optimizable models from PyTorch code. Any TorchScript program can be saved from a Python process and loaded in a process where there is no Python dependency.

Firstly run model_script.py to script python model and it can be load with both python and c++ without any python dependency

when you get scripted model you can use it with torchscript library in C++

brainUnet.cpp file gets path of scripted model  and input image and return predicted mask 

## results

### input image
![TCGA_FG_6689_20020326_29](https://user-images.githubusercontent.com/47300390/150556224-8c80d7ce-f536-4e78-ad86-08cb457a0a58.png)

### target mask

![TCGA_FG_6689_20020326_29_mask](https://user-images.githubusercontent.com/47300390/150556291-afb1fd7f-1627-4479-bca0-2d573a7e78f4.png)

### predicted mask

![predicted_mask](https://user-images.githubusercontent.com/47300390/150556327-363309f7-22de-4b3c-a65b-0c3e662f8378.png)






