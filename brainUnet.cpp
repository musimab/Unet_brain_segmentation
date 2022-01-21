
#include "brainUnet.hpp"

BrainUnetModel::BrainUnetModel(std::string& img_path, std::string& model_path) {

    m_input_img = cv::imread(img_path);
    
    if(m_input_img.empty()){
        cout << "Read frame failed" << endl;
    }

    m_module = torch::jit::load(model_path);
}

cv::Mat BrainUnetModel::forwardModel() {

    cv::cvtColor(m_input_img, m_rgb_img, cv::COLOR_BGR2RGB);
    torch::Tensor imgTensor = torch::from_blob(m_rgb_img.data, {m_rgb_img.rows, m_rgb_img.cols, 3}, torch::kByte );
    
    imgTensor = imgTensor.permute({2,0,1});
    imgTensor = imgTensor.toType(torch::kFloat);
    imgTensor = imgTensor.div(255);
    imgTensor = imgTensor.unsqueeze(0);
    imgTensor = imgTensor.to(torch::kCUDA);

    cout<< "input tensor shape:" << imgTensor.sizes() << endl;
    auto preds = m_module.forward({imgTensor}).toTensor();
    cout << "prediction shape:" << preds.sizes() << endl;

    m_predicted_mask =  toOpencvImage(preds);

    applyMaskRoi(m_input_img, m_predicted_mask);
    
    showCountersOfTumor(std::make_tuple(m_input_img, m_predicted_mask));

    return m_predicted_mask;
}


cv::Mat BrainUnetModel::toOpencvImage(torch::Tensor& pred_img) {
    
    pred_img = pred_img.mul(255).to(torch::kCPU);
    pred_img = pred_img.squeeze(0);
    pred_img = pred_img.permute({2, 1 ,0 });

    int width = pred_img.sizes()[0];
    int height = pred_img.sizes()[1];

    cv::Mat mask_img(cv::Size{height, width}, CV_32FC1, pred_img.data_ptr<float>());
    
    // Convert CV3_32_FC1 to CV_8UC1 type
    cv::Mat Temp = cv::Mat(mask_img.size(),CV_8U);
    mask_img.convertTo(Temp,CV_8UC1);
    mask_img=Temp.clone();

    return mask_img;
}

void BrainUnetModel::applyMaskRoi(cv::Mat& original_img, cv::Mat& predicted_mask) {
    
    cout<<"predicted_mask size:" << predicted_mask.size() <<" Ch:"<< predicted_mask.channels() << " "<< predicted_mask.type()<< endl;
    cout<< "original image size:" << original_img.size()<<" Ch:"<< original_img.channels() << " " << original_img.type() <<endl;

    cv::Mat predicted_mask_bgr, output;

    cv::cvtColor(predicted_mask , predicted_mask_bgr, cv::COLOR_GRAY2BGR);
    cout<< "predicted_mask_bgr size:" << predicted_mask_bgr.size() << " Ch:" << predicted_mask_bgr.channels() <<" " << predicted_mask_bgr.type() <<endl;

    cv::bitwise_and(original_img, predicted_mask_bgr, output);
    cv::imshow("masked region", output);

}


void BrainUnetModel::showCountersOfTumor(std::tuple<cv::Mat, cv::Mat>rgb_and_mask_image) const {
    
    auto [rgb, mask] = rgb_and_mask_image;

    std::vector<std::vector<cv::Point>> contours;
    cv::Mat contourOutput = mask.clone();
    cv::findContours(contourOutput, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat contourImage(contourOutput.size(), CV_8UC1, cv::Scalar(0,0,0));
    cv::Scalar colors;
    colors = cv::Scalar(200, 100, 200);

    for(size_t idx = 0; idx< contours.size(); idx++) {
        cv::drawContours(rgb, contours, idx, colors);
    }
    cv::imshow("Tumorous region", rgb);
    
}

