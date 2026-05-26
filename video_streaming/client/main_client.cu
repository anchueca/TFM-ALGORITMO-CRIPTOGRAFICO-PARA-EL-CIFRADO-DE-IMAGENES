/**
 * @file main_client.cu
 * @brief Encrypted video streaming client backend.
 *
 * Connects to the server's TCP stream, decrypts frames on GPU, and serves
 * the decrypted video as HTTP MJPEG on a local port. Controlled via
 * JSON commands on stdin (from the Flask wrapper).
 *
 * Protocol:
 *   stdin  → JSON commands:
 * {"action":"connect","host":"...","port":N,"password":"..."}
 *                           {"action":"disconnect"}
 *   MJPEG  → http://localhost:9090/stream  (decrypted video)
 */

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../common/stream_protocol.h"
#include "../common/video_crypto.cuh"

// ─── Globals ────────────────────────────────────────────────────────────────

static std::atomic<bool> g_running{true};
static std::atomic<bool> g_connected{false};
static std::atomic<bool> g_request_disconnect{false};

// Shared decrypted frame and JPEG buffer for MJPEG output
static std::mutex g_dec_mutex;
static cv::Mat g_decrypted_frame;
static std::vector<uchar> g_latest_jpeg;
static uint32_t g_dec_frame_id = 0;
static double g_fps = 0.0;
static bool g_new_dec_ready = false;

// Connection params (set via stdin command)
static std::mutex g_connect_mutex;
static std::string g_connect_host;
static int g_connect_port = 0;
static std::string g_connect_password;
static std::atomic<bool> g_request_connect{false};

// ─── Signal handler ─────────────────────────────────────────────────────────

void signal_handler(int) { g_running = false; }

// ─── Simple JSON parser (minimal, no external deps) ─────────────────────────

std::string json_get_string(const std::string &json, const std::string &key) {
  std::string search = "\"" + key + "\"";
  auto pos = json.find(search);
  if (pos == std::string::npos)
    return "";
  pos = json.find(':', pos);
  if (pos == std::string::npos)
    return "";
  pos = json.find('"', pos + 1);
  if (pos == std::string::npos)
    return "";
  auto end = json.find('"', pos + 1);
  if (end == std::string::npos)
    return "";
  return json.substr(pos + 1, end - pos - 1);
}

int json_get_int(const std::string &json, const std::string &key) {
  std::string search = "\"" + key + "\"";
  auto pos = json.find(search);
  if (pos == std::string::npos)
    return 0;
  pos = json.find(':', pos);
  if (pos == std::string::npos)
    return 0;
  // Skip whitespace
  pos++;
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t'))
    pos++;
  return std::stoi(json.substr(pos));
}

// ─── Decryption stream thread ───────────────────────────────────────────────

void decrypt_stream_thread() {
  while (g_running) {
    // Wait for connect request
    if (!g_request_connect) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }

    // Get connection parameters
    std::string host, password;
    int port;
    {
      std::lock_guard<std::mutex> lock(g_connect_mutex);
      host = g_connect_host;
      port = g_connect_port;
      password = g_connect_password;
      g_request_connect = false;
    }

    std::cerr << "[CLIENT] Connecting to " << host << ":" << port << "...\n";

    // Connect to server
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
      std::cerr << "[CLIENT] Socket creation failed\n";
      continue;
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr) <= 0) {
      std::cerr << "[CLIENT] Invalid address: " << host << "\n";
      close(sock);
      continue;
    }

    if (connect(sock, (sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
      std::cerr << "[CLIENT] Connection failed: " << strerror(errno) << "\n";
      close(sock);
      continue;
    }

    std::cerr << "[CLIENT] Connected! Receiving encrypted stream...\n";
    g_connected = true;
    g_request_disconnect = false;

    // Initialize decryptor after receiving first frame (need dimensions)
    VideoEncryptor *decryptor = nullptr;
    auto fps_start = std::chrono::steady_clock::now();
    int fps_count = 0;

    while (g_running && !g_request_disconnect) {
      FrameHeader hdr;
      if (!recv_frame_header(sock, hdr)) {
        std::cerr << "[CLIENT] Connection lost or closed by server\n";
        break;
      }

      // Allocate buffer and receive pixel data
      std::vector<unsigned char> pixel_data(hdr.data_size);
      if (!recv_all(sock, pixel_data.data(), hdr.data_size)) {
        std::cerr << "[CLIENT] Failed to receive frame data for ID "
                  << hdr.frame_id << "\n";
        break;
      }

      try {
        // Initialize decryptor on first frame
        if (!decryptor) {
          decryptor = new VideoEncryptor(password, hdr.width, hdr.height,
                                         hdr.channels, false);
          std::cerr << "[CLIENT] Decryptor initialized for " << hdr.width << "x"
                    << hdr.height << " (" << hdr.channels << " ch)\n";
        }

        int padded_dim = decryptor->getPaddedWidth();
        if (hdr.data_size != (uint32_t)(padded_dim * padded_dim)) {
          std::cerr << "[CLIENT] Size mismatch: expected "
                    << (padded_dim * padded_dim) << " but got " << hdr.data_size
                    << "\n";
          continue; // Skip this frame but don't disconnect
        }

        // Create cv::Mat from received data (single-channel padded square)
        cv::Mat encrypted(padded_dim, padded_dim, CV_8UC1, pixel_data.data());

        // Decrypt
        uint16_t frame_hash = static_cast<uint16_t>(hdr.image_hash);
        cv::Mat decrypted =
            decryptor->processFrame(encrypted.clone(), &frame_hash);

        // Encode as JPEG ONCE for broadcasting
        std::vector<uchar> jpeg_buf;
        std::vector<int> jpeg_params = {cv::IMWRITE_JPEG_QUALITY, 80};
        cv::imencode(".jpg", decrypted, jpeg_buf, jpeg_params);

        // Store for MJPEG output
        {
          std::lock_guard<std::mutex> lock(g_dec_mutex);
          g_decrypted_frame = decrypted;
          g_latest_jpeg = std::move(jpeg_buf);
          g_dec_frame_id = hdr.frame_id;
          g_new_dec_ready = true;
        }

        fps_count++;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - fps_start).count();
        if (elapsed >= 1.0) {
          g_fps = fps_count / elapsed;
          fps_count = 0;
          fps_start = now;
          std::cerr << "[CLIENT] Decrypting at " << g_fps
                    << " FPS (ID: " << g_dec_frame_id << ")\n";
        }
      } catch (const std::exception &e) {
        std::cerr << "[CLIENT] Error processing frame " << hdr.frame_id << ": "
                  << e.what() << "\n";
        // Don't break, try next frame
      }
    }

    // Cleanup
    close(sock);
    g_connected = false;
    delete decryptor;
    decryptor = nullptr;
    std::cerr << "[CLIENT] Disconnected.\n";
  }
}

// ─── MJPEG output server (serves decrypted video) ──────────────────────────

void mjpeg_client_handler(int client_fd) {
  const char *header =
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
      "Access-Control-Allow-Origin: *\r\n"
      "Cache-Control: no-cache\r\n"
      "Connection: close\r\n\r\n";

  if (send(client_fd, header, strlen(header), MSG_NOSIGNAL) <= 0) {
    close(client_fd);
    return;
  }

  uint32_t last_id = 0;

  while (g_running) {
    std::vector<uchar> jpeg_to_send;
    uint32_t current_id;

    {
      std::unique_lock<std::mutex> lock(g_dec_mutex);
      if (!g_new_dec_ready) {
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }
      jpeg_to_send = g_latest_jpeg; // Copying vector is fast enough
      current_id = g_dec_frame_id;
      g_new_dec_ready = false;
    }

    std::string part_header = "--frame\r\n"
                              "Content-Type: image/jpeg\r\n"
                              "Content-Length: " +
                              std::to_string(jpeg_to_send.size()) + "\r\n\r\n";

    if (send(client_fd, part_header.c_str(), part_header.size(),
             MSG_NOSIGNAL) <= 0)
      break;
    if (send(client_fd, jpeg_to_send.data(), jpeg_to_send.size(),
             MSG_NOSIGNAL) <= 0)
      break;
    if (send(client_fd, "\r\n", 2, MSG_NOSIGNAL) <= 0)
      break;
  }

  close(client_fd);
}

void mjpeg_server_thread(int port) {
  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) {
    perror("socket");
    return;
  }

  int opt = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);

  if (bind(server_fd, (sockaddr *)&addr, sizeof(addr)) < 0) {
    perror("bind MJPEG output");
    close(server_fd);
    return;
  }
  listen(server_fd, 5);
  std::cerr << "[MJPEG-OUT] Decrypted stream at http://localhost:" << port
            << "/stream\n";

  while (g_running) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(server_fd, &fds);
    timeval tv{1, 0};
    if (select(server_fd + 1, &fds, nullptr, nullptr, &tv) <= 0)
      continue;

    sockaddr_in client_addr{};
    socklen_t len = sizeof(client_addr);
    int client_fd = accept(server_fd, (sockaddr *)&client_addr, &len);
    if (client_fd < 0)
      continue;

    // Read HTTP request (discard)
    char req_buf[2048];
    recv(client_fd, req_buf, sizeof(req_buf), 0);

    std::thread(mjpeg_client_handler, client_fd).detach();
  }

  close(server_fd);
}

// ─── Status output thread (JSON on stdout for Flask) ────────────────────────

void status_output_thread() {
  while (g_running) {
    // Output status JSON every 500ms
    std::cout << "{\"connected\":" << (g_connected ? "true" : "false")
              << ",\"fps\":" << g_fps << ",\"frame_id\":" << g_dec_frame_id
              << "}" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
}

// ─── Stdin command reader ───────────────────────────────────────────────────

void stdin_reader_thread() {
  std::string line;
  while (g_running && std::getline(std::cin, line)) {
    if (line.empty())
      continue;

    std::string action = json_get_string(line, "action");

    if (action == "connect") {
      std::lock_guard<std::mutex> lock(g_connect_mutex);
      g_connect_host = json_get_string(line, "host");
      g_connect_port = json_get_int(line, "port");
      g_connect_password = json_get_string(line, "password");
      g_request_connect = true;
      std::cerr << "[CMD] Connect request: " << g_connect_host << ":"
                << g_connect_port << "\n";
    } else if (action == "disconnect") {
      g_request_disconnect = true;
      std::cerr << "[CMD] Disconnect request\n";
    } else if (action == "quit") {
      g_running = false;
    }
  }
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
  signal(SIGINT, signal_handler);
  signal(SIGPIPE, SIG_IGN);

  int mjpeg_port = 9090;
  if (argc > 1)
    mjpeg_port = std::stoi(argv[1]);

  std::cerr << "\n══════════════════════════════════════════════════\n"
            << "  ENCRYPTED VIDEO CLIENT BACKEND\n"
            << "  MJPEG output: http://localhost:" << mjpeg_port << "/stream\n"
            << "  Waiting for commands on stdin...\n"
            << "══════════════════════════════════════════════════\n\n";

  // Start threads
  std::thread mjpeg_thread(mjpeg_server_thread, mjpeg_port);
  std::thread decrypt_thread(decrypt_stream_thread);
  std::thread status_thread(status_output_thread);
  std::thread stdin_thread(stdin_reader_thread);

  stdin_thread.join();
  g_running = false;

  decrypt_thread.join();
  mjpeg_thread.join();
  status_thread.join();

  std::cerr << "[CLIENT] Stopped.\n";
  return 0;
}
