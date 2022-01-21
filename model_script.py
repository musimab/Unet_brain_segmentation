from pip import main
import torch
import cv2
import numpy as np
from torchvision import transforms
from matplotlib import pyplot as plt
import argparse

""" 
parser = argparse.ArgumentParser()

parser.add_argument("-M", "--model",
                        required=True,
                        help="Path to trained model (.pt file )")

parser.add_argument("-I", "--input", required= True,
        help="Input image for test")

parser.add_argument("-T", "--target", required= True,
        help="target image that model predict ")

args = vars(parser.parse_args())

unet_model = args["model"]
input_image= args["input"]
target_image= args["target"]
""" 

def saveScriptModel(model_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=True)
    
    model_script = torch.jit.script(model)
    #model_script.eval()
    model_script = model_script.to(device)

    # Save the scripted model file for subsequent use 
    torch.jit.save(model_script, model_path)

def loadDisplayModel(model_path):

    model_script = torch.jit.load(model_path)

    input_image = cv2.imread("data/TCGA_FG_6689_20020326_29.tif")
    target_mask = cv2.imread("data/TCGA_FG_6689_20020326_29_mask.tif")

    m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=m, std=s),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model_script = model_script.to('cuda')

    with torch.no_grad():
        output = model_script(input_batch)

    print(torch.round(output[0]))
    print(output.shape)

    mask = output[0].permute(1,2,0).to('cpu').numpy()
    mask = mask*255
    im_mask = mask.astype(np.uint8)
    predicted_mask = im_mask[:,:,0]
    # now the image can be plotted
    
    plt.imshow(predicted_mask , cmap="gray")
    plt.show()
    plt.imshow(target_mask, cmap="gray")
    plt.show()

if __name__ == '__main__':
    
    model_path = 'model/cuda_unet_brain.pt'
    saveScriptModel(model_path)
    loadDisplayModel(model_path)

    