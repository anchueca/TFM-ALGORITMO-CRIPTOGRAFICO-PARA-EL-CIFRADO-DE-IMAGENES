# ifndef ENCRYPTION_CUH
# define ENCRYPTION_CUH

// CUDA headers primero
#include <cuda_runtime.h>

// Standard headers
#include <algorithm>
#include <vector>
#include <iostream>

// Project headers
#include "kernels.cuh"
#include "aux.cuh"
#include "automata.cuh"
#include "encryption_aux.cuh"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace std;

void encrypt_image(cv::Mat image, const std::string& password, int rounds, int verbose);



# endif // ENCRYPTION_CUH