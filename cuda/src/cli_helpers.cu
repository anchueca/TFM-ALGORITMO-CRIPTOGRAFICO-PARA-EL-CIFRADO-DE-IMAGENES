#include "../include/cli_config.cuh"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

using namespace std;

void print_usage(const char *prog_name) {
  cout << "Usage: " << prog_name
       << " <InputPath> <OutputPath|SHOW|STDOUT> <Password> "
       << "<Rounds <Mode(1=Enc/0=Dec)> <BlockSize> "
       << "<AutoSteps> <TransLen> <chaosParam> <Verbose(0/1)>" << endl;
  cout << "\nOutput Options:" << endl;
  cout << "  <Path>   : Saves to file (e.g., output.tif)" << endl;
  cout << "  SHOW     : Opens a window with the result (Requires GUI)" << endl;
  cout << "  STDOUT   : Pipes binary image data (TIFF) to standard output"
       << endl;
}

bool parse_arguments(int argc, char **argv, AppConfig &config) {
  const size_t required_args = 11;
  if (argc != required_args) {
    cerr << "[ERROR] Invalid number of arguments! " << argc << " received, "
         << required_args << " required." << endl;
    return false;
  }

  try {
    config.input_image_path = argv[1];
    config.output_arg = argv[2];
    config.password = argv[3];
    config.params.rounds = stoi(argv[4]);
    config.encrypt = (strcmp(argv[5], "1") == 0);
    config.params.block_size = stoi(argv[6]);
    config.params.automata_steps = stoi(argv[7]);
    config.params.transition_length = stoi(argv[8]);
#ifdef USE_DOUBLE_PRECISION
    config.params.chaos_parameter = stod(argv[9]);
#else
    config.params.chaos_parameter = stof(argv[9]);
#endif
    config.verbose = (strcmp(argv[10], "1") == 0);
    config.params.num_extra_seeds = 1; // Const

    if (config.output_arg == "SHOW" || config.output_arg == "NULL") {
      config.output_mode = OutputMode::DISPLAY_WINDOW;
    } else if (config.output_arg == "STDOUT") {
      config.output_mode = OutputMode::STDOUT_STREAM;
      config.verbose = false; // Disable verbose if piping to stdout
    } else {
      config.output_mode = OutputMode::FILE_SAVE;
    }
  } catch (const std::exception &e) {
    cerr << "[ERROR] Parsing arguments failed: " << e.what() << endl;
    return false;
  }
  return true;
}

cv::Mat load_image(const string &path) {
  cv::Mat image;
  if (path == "STDIN") {
    std::ios::sync_with_stdio(false);
    std::vector<uchar> input_buffer((std::istreambuf_iterator<char>(std::cin)),
                                    (std::istreambuf_iterator<char>()));
    if (input_buffer.empty()) {
      cerr << "[ERROR] STDIN mode selected but no data received." << endl;
      return image;
    }
    try {
      image = cv::imdecode(input_buffer, cv::IMREAD_UNCHANGED);
    } catch (const cv::Exception &e) {
      cerr << "[ERROR] Failed to decode image from STDIN: " << e.what() << endl;
    }
  } else {
    image = cv::imread(path, cv::IMREAD_UNCHANGED);
  }
  return image;
}

void print_initial_report(const AppConfig &config, const cv::Mat &image) {
  if (!config.verbose)
    return;

  std::cout << "\n============================================================"
            << std::endl;
  std::cout << "                     CLI EXECUTION START                      "
            << std::endl;
  std::cout << "============================================================"
            << std::endl;
  std::cout << "  Input File:    " << config.input_image_path << std::endl;
  std::cout << "  Output Mode:   " << config.output_arg << std::endl;
  std::cout << "  Loaded Size:   " << image.cols << "x" << image.rows
            << std::endl;
  std::cout << "  Channels:      " << image.channels() << std::endl;
  std::cout << "============================================================\n"
            << std::endl;

#ifdef USE_DOUBLE_PRECISION
  std::cout << " [PRECISION INFO]:       DOUBLE (High Precision)" << std::endl;
#else
  std::cout << " [PRECISION INFO]:       FLOAT (Standard Precision)"
            << std::endl;
#endif
}

bool handle_output(OutputMode mode, const string &output_arg,
                   const cv::Mat &image, bool verbose) {
  if (image.empty()) {
    cerr << "[ERROR] Resulting image is empty. Encryption/Decryption failed."
         << endl;
    return false;
  }

  switch (mode) {
  case OutputMode::FILE_SAVE: {
    if (!cv::imwrite(output_arg, image)) {
      cerr << "[ERROR] Failed to write output image to: " << output_arg << endl;
      return false;
    }
    if (verbose)
      std::cout << "Image saved successfully to " << output_arg << std::endl;
    break;
  }
  case OutputMode::DISPLAY_WINDOW: {
    try {
      cv::namedWindow("Cipher Result", cv::WINDOW_AUTOSIZE);
      cv::imshow("Cipher Result", image);
      if (verbose)
        std::cout << "Displaying image. Press any key to close..." << std::endl;
      cv::waitKey(0);
    } catch (const cv::Exception &e) {
      cerr << "[ERROR] GUI Call failed (No X11/Display?): " << e.what() << endl;
      return false;
    }
    break;
  }
  case OutputMode::STDOUT_STREAM: {
    std::vector<uchar> buf;
    try {
      if (cv::imencode(".tif", image, buf)) {
        std::cout.write(reinterpret_cast<const char *>(buf.data()), buf.size());
        std::cout.flush();
      } else {
        cerr << "[ERROR] Failed to encode image for streaming." << endl;
        return false;
      }
    } catch (const cv::Exception &e) {
      cerr << "[ERROR] Encoding for stream failed: " << e.what() << endl;
      return false;
    }
    break;
  }
  }
  return true;
}
