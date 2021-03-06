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

'
 mkdir build
     cd build
       cmake ..
       
' 

## results

### input image

![TCGA_FG_6689_20020326_29](https://user-images.githubusercontent.com/47300390/150651624-1e05a346-cd83-40ad-a8c3-228365d12db5.png)
### target mask

![TCGA_FG_6689_20020326_29_mask](https://user-images.githubusercontent.com/47300390/150651627-f6f0f713-c437-4054-ad18-f1ea4f8cc188.png)


### predicted mask

![predicted_mask](https://user-images.githubusercontent.com/47300390/150651633-5465ff44-27d6-4122-89a5-ff287d4349f5.png)









