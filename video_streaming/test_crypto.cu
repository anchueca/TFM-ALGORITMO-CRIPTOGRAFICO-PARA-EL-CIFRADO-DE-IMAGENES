#include <opencv2/opencv.hpp>
#include <iostream>
#include "common/video_crypto.cuh"

int main() {
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::randu(frame, cv::Scalar(0,0,0), cv::Scalar(255,255,255));
    
    std::cout << "Creating Encryptor...\n";
    VideoEncryptor enc("pepe", 640, 480, 3, true);
    
    std::cout << "Encrypting...\n";
    cv::Mat encrypted = enc.processFrame(frame);
    
    std::cout << "Encrypted size: " << encrypted.cols << "x" << encrypted.rows << "\n";
    
    std::cout << "Creating Decryptor...\n";
    VideoEncryptor dec("pepe", 640, 480, 3, false);
    
    std::cout << "Decrypting...\n";
    cv::Mat decrypted = dec.processFrame(encrypted.clone());
    
    std::cout << "Decrypted size: " << decrypted.cols << "x" << decrypted.rows << "\n";
    
    double diff = cv::norm(frame, decrypted, cv::NORM_L1);
    std::cout << "Difference: " << diff << "\n";
    return 0;
}
