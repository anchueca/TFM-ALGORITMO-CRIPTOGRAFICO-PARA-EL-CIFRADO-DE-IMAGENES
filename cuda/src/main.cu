#include "../include/encryption.cuh"

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

enum class OutputMode {
    FILE_SAVE,
    DISPLAY_WINDOW,
    STDOUT_STREAM
};

void print_usage(const char *prog_name) {
  cout << "Usage: " << prog_name << " <InputPath> <OutputPath|SHOW|STDOUT> <Password> "
       << "<Rounds> <Verbose(0/1)> <Mode(1=Enc/0=Dec)> "
       << "<BlockSize> <Precision> <AutoSteps> <TransLen>" << endl;
  cout << "\nOutput Options:" << endl;
  cout << "  <Path>   : Saves to file (e.g., output.tif)" << endl;
  cout << "  SHOW     : Opens a window with the result (Requires GUI)" << endl;
  cout << "  STDOUT   : Pipes binary image data (TIFF) to standard output" << endl;
}

int main(int argc, char **argv) {
  const size_t required_args = 11;

  if (argc != required_args) {
    cerr << "[ERROR] Invalid number of arguments! " << argc << " received, "
         << required_args << " required." << endl;
    print_usage(argv[0]);
    return -1;
  }

  EncryptionParams params;
  string input_image_path;
  string output_arg;
  string password;
  bool verbose_arg = false;
  bool encrypt = true;
  OutputMode output_mode = OutputMode::FILE_SAVE;

  // --- 1. Parse Arguments ---
  try {
    input_image_path = argv[1];
    output_arg = argv[2];
    password = argv[3];
    params.rounds = stoi(argv[4]);

    verbose_arg = (strcmp(argv[5], "1") == 0);
    encrypt = (strcmp(argv[6], "1") == 0);

    params.block_size = stoi(argv[7]);
    params.precision_level = stoi(argv[8]);
    params.automata_steps = stoi(argv[9]);
    params.transition_length = stoi(argv[10]);

    if (output_arg == "SHOW" || output_arg == "NULL") {
        output_mode = OutputMode::DISPLAY_WINDOW;
    } else if (output_arg == "STDOUT") {
        output_mode = OutputMode::STDOUT_STREAM;
    } else {
        output_mode = OutputMode::FILE_SAVE;
    }

  } catch (const std::exception &e) {
    cerr << "[ERROR] Parsing arguments failed: " << e.what() << endl;
    print_usage(argv[0]);
    return -1;
  }

  if (output_arg == "SHOW" || output_arg == "NULL") {
        output_mode = OutputMode::DISPLAY_WINDOW;
    } else if (output_arg == "STDOUT") {
        output_mode = OutputMode::STDOUT_STREAM;
        verbose_arg = false;
    } else {
        output_mode = OutputMode::FILE_SAVE;
    }

  // --- 2. Load Image ---
  cv::Mat image;
  if (input_image_path == "STDIN") {
      // MODO MEMORIA: Leemos los bytes crudos desde la entrada estándar (Pipe)
      
      // Desactivar sincronización para velocidad (opcional pero recomendado)
      std::ios::sync_with_stdio(false);
      
      // Leemos todo el flujo de entrada en un vector de bytes
      // istreambuf_iterator lee char por char hasta el final del stream (EOF)
      std::vector<uchar> input_buffer(
          (std::istreambuf_iterator<char>(std::cin)),
          (std::istreambuf_iterator<char>())
      );

      if (input_buffer.empty()) {
          cerr << "[ERROR] STDIN mode selected but no data received." << endl;
          return -1;
      }

      // Decodificamos el buffer directamente a una Matriz OpenCV
      try {
          image = cv::imdecode(input_buffer, cv::IMREAD_UNCHANGED);
      } catch (const cv::Exception& e) {
          cerr << "[ERROR] Failed to decode image from STDIN: " << e.what() << endl;
          return -1;
      }

  } else {
      // MODO DISCO: Comportamiento clásico
      image = cv::imread(input_image_path, cv::IMREAD_UNCHANGED);
  }

  if (image.empty()) {
      cerr << "[ERROR] Image data is empty or corrupted." << endl;
      return -1;
  }

  // --- 3. Initial Report ---
  if (verbose_arg) {
    std::cout
        << "\n============================================================"
        << std::endl;
    std::cout << "                     CLI EXECUTION START                      "
              << std::endl;
    std::cout << "============================================================"
              << std::endl;
    std::cout << "  Input File:    " << input_image_path << std::endl;
    std::cout << "  Output Mode:   " << output_arg << std::endl;
    std::cout << "  Loaded Size:   " << image.cols << "x" << image.rows
              << std::endl;
    std::cout << "  Channels:      " << image.channels() << std::endl;
    std::cout
        << "============================================================\n"
        << std::endl;
  }

  // --- 4. Execution ---
  auto start = std::chrono::high_resolution_clock::now();

  try {
    // Call the smart encryption function
    encrypt_image(image, password, params, verbose_arg, encrypt);
  } catch (const std::exception &e) {
    cerr << "\n[FATAL ERROR] During encryption process: " << e.what() << endl;
    return -1;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;

  if (verbose_arg) {
    std::cout << "Total Pipeline Time: " << time.count() << " s" << std::endl;
  }

  // --- 5. Validation & Output Handling ---
  if (image.empty()) {
    cerr << "[ERROR] Resulting image is empty. Encryption/Decryption failed."
         << endl;
    return -1;
  }

  switch (output_mode) {
      case OutputMode::FILE_SAVE: {
          if (!cv::imwrite(output_arg, image)) {
            cerr << "[ERROR] Failed to write output image to: " << output_arg
                 << endl;
            return -1;
          }
          if (verbose_arg) std::cout << "Image saved successfully to " << output_arg << std::endl;
          break;
      }

      case OutputMode::DISPLAY_WINDOW: {
          try {
              cv::namedWindow("Cipher Result", cv::WINDOW_AUTOSIZE);
              cv::imshow("Cipher Result", image);
              if (verbose_arg) std::cout << "Displaying image. Press any key to close..." << std::endl;
              cv::waitKey(0); // Espera infinita hasta pulsar tecla
          } catch (const cv::Exception& e) {
              cerr << "[ERROR] GUI Call failed (No X11/Display?): " << e.what() << endl;
              return -1;
          }
          break;
      }

      case OutputMode::STDOUT_STREAM: {
          std::vector<uchar> buf;
          bool success = false;
          try {
             success = cv::imencode(".tif", image, buf);
          } catch (const cv::Exception& e) {
             cerr << "[ERROR] Encoding for stream failed: " << e.what() << endl;
             return -1;
          }

          if (success) {
              std::cout.write(reinterpret_cast<const char*>(buf.data()), buf.size());
              std::cout.flush();
          } else {
              cerr << "[ERROR] Failed to encode image for streaming." << endl;
              return -1;
          }
          break;
      }
  }

  return 0;
}