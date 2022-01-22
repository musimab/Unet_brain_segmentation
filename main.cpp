#include<iostream>
#include<vector>
#include<torch/script.h>
#include<opencv2/opencv.hpp>
#include<time.h>
#include<map>
#include "brainUnet.hpp"

using std::cout;
using std::endl;

int main() {

    std::string module_path = "/home/mustafa/Desktop/pytorch_cpp/Unet_CPP/model/cuda_unet_brain.pt";
    std::string img_path = "/home/mustafa/Desktop/pytorch_cpp/Unet_CPP/data/TCGA_FG_6689_20020326_29.tif";

    cv::Mat target_mask = cv::imread("/home/mustafa/Desktop/pytorch_cpp/Unet_CPP/data/TCGA_FG_6689_20020326_29_mask.tif");

    if(target_mask.empty()){
        cout << "Unable to read frame" << endl;
        return 0;
    }
    
    BrainUnetModel& UnetBrain = BrainUnetModel::getInstance();
    UnetBrain.setDataPath(img_path);
    UnetBrain.setModelPath(module_path);
    cv::Mat output_mask = UnetBrain.forwardModel();
    
    cv::imshow("target_mask", target_mask);
    cv::imshow("predicted_mask", output_mask);
    cv::imwrite("../predicted_mask.tif", output_mask);
    cv::waitKey(0);
 


    






    



    return 0;
}
