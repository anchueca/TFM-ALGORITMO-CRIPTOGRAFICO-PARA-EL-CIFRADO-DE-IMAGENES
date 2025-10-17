#include "../include/encryption.cuh"

#include<iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

int main(int argc, char** argv) {
    if (argc != 6){
        cerr << "Error"<<endl;
        return -1;
    }

    string input_image_path = argv[1];
    string password = argv[2];
    int rounds = stoi(argv[3]);
    string output_image_path = argv[4];
    int verbose = stoi(argv[5]);

    cv::Mat image = cv::imread(input_image_path);
    if (image.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }

    int channels = image.channels();

    if(channels !=1) image= unstack_image(image);

    encrypt_image(image, password, rounds, verbose);

    if(channels !=1) image= stack_image(image);

    if (image.empty()) {
        cerr << "Encryption failed!" << endl;
        return -1;
    }

    //cv::imshow("Encrypted Image", image);
    //cv::waitKey(0);
    cv::imwrite(output_image_path,image);

    return 0;
}