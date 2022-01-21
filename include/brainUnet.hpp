#pragma once 

#include<iostream>
#include<vector>
#include<torch/script.h>
#include<opencv2/opencv.hpp>
#include<time.h>
#include<map>

using std::cout;
using std::endl;

class BrainUnetModel{

    public:
    BrainUnetModel(std::string& img_path, std::string& model_path);
    cv::Mat toOpencvImage(torch::Tensor& pred_img);
    void applyMaskRoi(cv::Mat& original_img,cv::Mat& predicted_mask);
    void showCountersOfTumor(std::tuple<cv::Mat, cv::Mat>rgb_and_mask_image) const;
    cv::Mat forwardModel();
    
    private:
    cv::Mat m_input_img;
    cv::Mat m_rgb_img;
    cv::Mat m_predicted_mask;
    torch::jit::Module m_module;

};
