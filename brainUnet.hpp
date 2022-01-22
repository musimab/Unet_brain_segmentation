#pragma once 

#include<iostream>
#include<vector>
#include<torch/script.h>
#include<opencv2/opencv.hpp>
#include<time.h>
#include<map>

using std::cout;
using std::endl;

//Meyers Singleton pattern
class BrainUnetModel{

    public:
    static BrainUnetModel& getInstance();
    
    cv::Mat toOpencvImage(torch::Tensor& pred_img);
    void applyMaskRoi(cv::Mat& original_img,cv::Mat& predicted_mask) const;
    void showCountersOfTumor(std::tuple<cv::Mat, cv::Mat>rgb_and_mask_image) const;
    cv::Mat forwardModel();
    void setModelPath(std::string& model_path);
    void setDataPath(std::string& data_path);

    private:
    ~BrainUnetModel()= default;
    BrainUnetModel() = default;
    BrainUnetModel(const BrainUnetModel&) = delete;
    BrainUnetModel& operator = (const BrainUnetModel&) = delete;
    
    cv::Mat m_input_img;
    cv::Mat m_rgb_img;
    cv::Mat m_predicted_mask;
    torch::jit::Module m_module;

};
