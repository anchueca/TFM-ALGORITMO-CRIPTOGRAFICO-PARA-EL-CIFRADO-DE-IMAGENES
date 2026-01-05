#ifndef CLI_CONFIG_CUH
#define CLI_CONFIG_CUH

#include "structs.cuh"
#include <opencv2/core.hpp>
#include <string>

enum class OutputMode { FILE_SAVE, DISPLAY_WINDOW, STDOUT_STREAM };

struct AppConfig {
  EncryptionParams params;
  std::string input_image_path;
  std::string output_arg;
  std::string password;
  bool verbose;
  bool encrypt;
  bool use_raw_key;
  OutputMode output_mode;
};

void print_usage(const char *prog_name);

bool parse_arguments(int argc, char **argv, AppConfig &config);

cv::Mat load_image(const std::string &path);

void print_initial_report(const AppConfig &config, const cv::Mat &image);

bool handle_output(OutputMode mode, const std::string &output_arg,
                   const cv::Mat &image, bool verbose);

#endif // CLI_CONFIG_CUH
