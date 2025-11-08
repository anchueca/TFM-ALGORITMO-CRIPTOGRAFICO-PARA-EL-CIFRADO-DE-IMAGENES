# ifndef ENCRYPTION_CUH
# define ENCRYPTION_CUH

#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace std;

struct EncryptionParams {
    size_t rounds;
    size_t block_size;
    size_t precision_level;
    size_t automata_steps;
    size_t transition_length;
};

void encrypt_image(cv::Mat image, const std::string& password, const EncryptionParams& params, bool verbose, bool encrypt);
void encryption_process(unsigned char** d_image, unsigned char** d_image_out, unsigned int* d_permutation_rows, unsigned int* d_permutation_cols, unsigned int* d_permutation_blocks, size_t cols, size_t rows, std::vector<unsigned char> flow_seeds, size_t block_size, size_t rounds,bool verbose);
void unencryption_process(unsigned char** d_image, unsigned char** d_image_out, unsigned int* d_permutation_rows, unsigned int* d_permutation_cols, unsigned int* d_permutation_blocks, size_t cols, size_t rows, std::vector<unsigned char> flow_seeds, size_t block_size, size_t rounds);


# endif // ENCRYPTION_CUH