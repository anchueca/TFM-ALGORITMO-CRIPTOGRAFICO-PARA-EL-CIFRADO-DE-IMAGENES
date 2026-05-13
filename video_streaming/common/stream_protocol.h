#ifndef STREAM_PROTOCOL_H
#define STREAM_PROTOCOL_H

#include <arpa/inet.h>
#include <cstdint>
#include <cstring>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

// Magic number to identify our frame protocol
#define FRAME_MAGIC 0xCF12CF12

/**
 * @brief Header sent before each frame over TCP.
 *
 * All fields are in network byte order (big-endian).
 */
struct FrameHeader {
  uint32_t magic;     // FRAME_MAGIC
  uint32_t width;     // Frame width in pixels
  uint32_t height;    // Frame height in pixels
  uint32_t channels;  // Number of channels (1 or 3)
  uint32_t data_size; // Size of pixel data in bytes
  uint32_t frame_id;  // Monotonic frame counter
  uint32_t image_hash; // 16-bit hash (stored as 32-bit for alignment/future)
};

static const size_t FRAME_HEADER_SIZE = sizeof(FrameHeader);

/**
 * @brief Send exactly n bytes over a socket.
 * @return true on success, false on error/disconnect.
 */
inline bool send_all(int sock, const void *data, size_t len) {
  const char *ptr = static_cast<const char *>(data);
  size_t remaining = len;
  while (remaining > 0) {
    ssize_t sent = send(sock, ptr, remaining, MSG_NOSIGNAL);
    if (sent <= 0)
      return false;
    ptr += sent;
    remaining -= sent;
  }
  return true;
}

/**
 * @brief Receive exactly n bytes from a socket.
 * @return true on success, false on error/disconnect.
 */
inline bool recv_all(int sock, void *data, size_t len) {
  char *ptr = static_cast<char *>(data);
  size_t remaining = len;
  while (remaining > 0) {
    ssize_t received = recv(sock, ptr, remaining, 0);
    if (received <= 0)
      return false;
    ptr += received;
    remaining -= received;
  }
  return true;
}

/**
 * @brief Send a frame with its header over TCP.
 */
inline bool send_frame(int sock, uint32_t width, uint32_t height,
                       uint32_t channels, uint32_t data_size,
                       const unsigned char *data, uint32_t frame_id,
                       uint32_t image_hash) {
  FrameHeader hdr;
  hdr.magic = htonl(FRAME_MAGIC);
  hdr.width = htonl(width);
  hdr.height = htonl(height);
  hdr.channels = htonl(channels);
  hdr.data_size = htonl(data_size);
  hdr.frame_id = htonl(frame_id);
  hdr.image_hash = htonl(image_hash);

  if (!send_all(sock, &hdr, FRAME_HEADER_SIZE))
    return false;
  if (!send_all(sock, data, data_size))
    return false;
  return true;
}

/**
 * @brief Receive a frame header and validate magic.
 */
inline bool recv_frame_header(int sock, FrameHeader &hdr) {
  if (!recv_all(sock, &hdr, FRAME_HEADER_SIZE))
    return false;
  hdr.magic = ntohl(hdr.magic);
  hdr.width = ntohl(hdr.width);
  hdr.height = ntohl(hdr.height);
  hdr.channels = ntohl(hdr.channels);
  hdr.data_size = ntohl(hdr.data_size);
  hdr.frame_id = ntohl(hdr.frame_id);
  hdr.image_hash = ntohl(hdr.image_hash);
  return hdr.magic == FRAME_MAGIC;
}

#endif // STREAM_PROTOCOL_H
