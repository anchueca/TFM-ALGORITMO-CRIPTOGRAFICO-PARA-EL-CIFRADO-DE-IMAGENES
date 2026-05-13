/**
 * @file main_server.cu
 * @brief Encrypted video streaming server.
 *
 * Captures webcam frames, encrypts them on the GPU using the chaotic cipher,
 * and streams them via:
 *   - TCP raw protocol (port --port)   → for the custom client (lossless)
 *   - HTTP MJPEG      (port --mjpeg-port) → for VLC viewing (encrypted noise)
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
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "../common/stream_protocol.h"
#include "../common/video_crypto.cuh"

// ─── Globals ────────────────────────────────────────────────────────────────

static std::atomic<bool> g_running{true};

// Shared latest encrypted frame and JPEG buffer (for broadcasting)
static std::mutex g_broadcast_mutex;
static cv::Mat g_latest_encrypted;
static std::vector<uchar> g_latest_jpeg;
static uint32_t g_latest_frame_id = 0;
static uint16_t g_latest_image_hash = 0;
static bool g_new_frame_ready = false;
static int g_actual_w, g_actual_h, g_padded_w, g_padded_h;

// ─── Signal handler ─────────────────────────────────────────────────────────

void signal_handler(int) { g_running = false; }

// ─── CLI parsing ────────────────────────────────────────────────────────────

struct ServerConfig {
    std::string password;
    int port       = 8554;
    int mjpeg_port = 8555;
    int device     = 0;
    int width      = 640;
    int height     = 480;
};

bool parse_args(int argc, char** argv, ServerConfig& cfg) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--password" && i + 1 < argc) cfg.password = argv[++i];
        else if (arg == "--port" && i + 1 < argc) cfg.port = std::stoi(argv[++i]);
        else if (arg == "--mjpeg-port" && i + 1 < argc) cfg.mjpeg_port = std::stoi(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) cfg.device = std::stoi(argv[++i]);
        else if (arg == "--resolution" && i + 1 < argc) {
            std::string res = argv[++i];
            auto x = res.find('x');
            if (x != std::string::npos) {
                cfg.width  = std::stoi(res.substr(0, x));
                cfg.height = std::stoi(res.substr(x + 1));
            }
        }
        else if (arg == "--help") {
            std::cout << "Usage: video_server --password <KEY> [options]\n"
                      << "  --port <N>         TCP port for client (default 8554)\n"
                      << "  --mjpeg-port <N>   HTTP MJPEG port for VLC (default 8555)\n"
                      << "  --device <N>       Webcam device index (default 0)\n"
                      << "  --resolution WxH   Capture resolution (default 640x480)\n";
            return false;
        }
    }
    if (cfg.password.empty()) {
        std::cerr << "[ERROR] --password is required.\n";
        return false;
    }
    return true;
}

// ─── TCP Client Handler (raw frame protocol) ───────────────────────────────

static std::mutex g_tcp_clients_mutex;
static std::vector<int> g_tcp_clients;

void tcp_accept_thread(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind TCP"); close(server_fd); return;
    }
    listen(server_fd, 5);

    std::cerr << "[TCP] Listening on port " << port << std::endl;

    while (g_running) {
        sockaddr_in client_addr{};
        socklen_t len = sizeof(client_addr);

        // Use select with timeout to allow clean shutdown
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(server_fd, &fds);
        timeval tv{1, 0};
        if (select(server_fd + 1, &fds, nullptr, nullptr, &tv) <= 0) continue;

        int client_fd = accept(server_fd, (sockaddr*)&client_addr, &len);
        if (client_fd < 0) continue;

        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
        std::cerr << "[TCP] Client connected: " << client_ip << std::endl;

        std::lock_guard<std::mutex> lock(g_tcp_clients_mutex);
        g_tcp_clients.push_back(client_fd);
    }

    close(server_fd);
}

void tcp_broadcast_thread() {
    uint32_t last_sent_id = 0xFFFFFFFF;
    
    while (g_running) {
        cv::Mat frame_to_send;
        uint32_t frame_id;
        uint16_t image_hash;
        int aw, ah, pw, ph;
        
        {
            std::unique_lock<std::mutex> lock(g_broadcast_mutex);
            if (!g_new_frame_ready || g_latest_frame_id == last_sent_id) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            frame_to_send = g_latest_encrypted.clone();
            frame_id = g_latest_frame_id;
            image_hash = g_latest_image_hash;
            aw = g_actual_w; ah = g_actual_h;
            pw = g_padded_w; ph = g_padded_h;
            last_sent_id = frame_id;
        }

        std::lock_guard<std::mutex> clients_lock(g_tcp_clients_mutex);
        if (g_tcp_clients.empty()) continue;

        uint32_t padded_size = pw * ph;
        std::vector<int> alive;
        for (int fd : g_tcp_clients) {
            // Use a short timeout for send if possible, but for now just send
             if (send_frame(fd, aw, ah, 3, padded_size,
                             frame_to_send.data, frame_id, image_hash)) {
                alive.push_back(fd);
            } else {
                std::cerr << "[TCP] Client disconnected (fd=" << fd << ")\n";
                close(fd);
            }
        }
        g_tcp_clients = alive;
    }
}

// ─── MJPEG HTTP Server (for VLC) ───────────────────────────────────────────

void mjpeg_client_handler(int client_fd) {
    // Send HTTP header
    const char* header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n\r\n";

    if (send(client_fd, header, strlen(header), MSG_NOSIGNAL) <= 0) {
        close(client_fd);
        return;
    }

    uint32_t last_id = 0xFFFFFFFF;

    while (g_running) {
        std::vector<uchar> jpeg_buf;
        uint32_t current_id;
        
        {
            std::unique_lock<std::mutex> lock(g_broadcast_mutex);
            if (!g_new_frame_ready || g_latest_frame_id == last_id) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            jpeg_buf = g_latest_jpeg;
            current_id = g_latest_frame_id;
        }

        if (jpeg_buf.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        // Send multipart boundary + JPEG
        std::string part_header =
            "--frame\r\nContent-Type: image/jpeg\r\n"
            "Content-Length: " + std::to_string(jpeg_buf.size()) + "\r\n\r\n";

        if (send(client_fd, part_header.c_str(), part_header.size(),
                 MSG_NOSIGNAL) <= 0) break;
        if (send(client_fd, jpeg_buf.data(), jpeg_buf.size(), MSG_NOSIGNAL) <= 0) break;
        if (send(client_fd, "\r\n", 2, MSG_NOSIGNAL) <= 0) break;
        
        last_id = current_id;
    }

    close(client_fd);
}

void mjpeg_server_thread(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind MJPEG"); close(server_fd); return;
    }
    listen(server_fd, 5);

    std::cerr << "[MJPEG] VLC stream available at http://0.0.0.0:"
              << port << "/" << std::endl;

    while (g_running) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(server_fd, &fds);
        timeval tv{1, 0};
        if (select(server_fd + 1, &fds, nullptr, nullptr, &tv) <= 0) continue;

        sockaddr_in client_addr{};
        socklen_t len = sizeof(client_addr);
        int client_fd = accept(server_fd, (sockaddr*)&client_addr, &len);
        if (client_fd < 0) continue;

        // Read and discard the HTTP request
        char req_buf[2048];
        recv(client_fd, req_buf, sizeof(req_buf), 0);

        std::cerr << "[MJPEG] VLC client connected\n";
        std::thread(mjpeg_client_handler, client_fd).detach();
    }

    close(server_fd);
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    signal(SIGPIPE, SIG_IGN);

    ServerConfig cfg;
    if (!parse_args(argc, argv, cfg)) return 1;

    // ── Open webcam ──
    cv::VideoCapture cap(cfg.device);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open webcam device " << cfg.device << "\n";
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

    int actual_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int actual_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cerr << "[SERVER] Webcam opened: " << actual_w << "x" << actual_h
              << std::endl;

    // ── Initialize encryptor ──
    VideoEncryptor encryptor(cfg.password, actual_w, actual_h, 3, true);

    // ── Start network threads ──
    std::thread tcp_thread(tcp_accept_thread, cfg.port);
    std::thread tcp_broadcast_thr(tcp_broadcast_thread);
    std::thread mjpeg_thread(mjpeg_server_thread, cfg.mjpeg_port);

    std::cerr << "\n══════════════════════════════════════════════════\n"
              << "  ENCRYPTED VIDEO SERVER RUNNING\n"
              << "  TCP (client):  port " << cfg.port << "\n"
              << "  MJPEG (VLC):   http://0.0.0.0:" << cfg.mjpeg_port << "/\n"
              << "  Press Ctrl+C to stop.\n"
              << "══════════════════════════════════════════════════\n\n";

    // ── Main capture + encrypt loop ──
    cv::Mat frame;
    uint32_t frame_counter = 0;
    auto fps_start = std::chrono::steady_clock::now();
    int fps_count = 0;

    while (g_running) {
        cap >> frame;
        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Encode as JPEG ONCE for all MJPEG clients
        // Reduce quality to 70 for preview, it's just noise anyway
        std::vector<uchar> jpeg_buf;
        std::vector<int> jpeg_params = {cv::IMWRITE_JPEG_QUALITY, 70};
        // ── 3. Encrypt ──
        uint16_t frame_hash = 0;
        cv::Mat encrypted = encryptor.processFrame(frame, &frame_hash);

        // ── 3b. Encode for MJPEG ──
        cv::imencode(".jpg", encrypted, jpeg_buf, jpeg_params);

        // ── 4. Save results for broadcasting ──
        {
            std::lock_guard<std::mutex> lock(g_broadcast_mutex);
            g_latest_encrypted = encrypted.clone();
            g_latest_image_hash = frame_hash;
            g_latest_jpeg = std::move(jpeg_buf);
            g_latest_frame_id = frame_counter;
            g_actual_w = actual_w;
            g_actual_h = actual_h;
            g_padded_w = encryptor.getPaddedWidth();
            g_padded_h = encryptor.getPaddedHeight();
            g_new_frame_ready = true;
        }

        frame_counter++;
        fps_count++;

        // Print FPS every 2 seconds
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - fps_start).count();
        if (elapsed >= 2.0) {
            std::cerr << "[SERVER] FPS: " << (fps_count / elapsed)
                      << " | Frames: " << frame_counter << "\r" << std::flush;
            fps_count = 0;
            fps_start = now;
        }
    }

    std::cerr << "\n[SERVER] Shutting down...\n";
    cap.release();

    // Close all TCP clients
    {
        std::lock_guard<std::mutex> lock(g_tcp_clients_mutex);
        for (int fd : g_tcp_clients) close(fd);
        g_tcp_clients.clear();
    }

    tcp_thread.join();
    tcp_broadcast_thr.join();
    mjpeg_thread.join();

    std::cerr << "[SERVER] Stopped.\n";
    return 0;
}
