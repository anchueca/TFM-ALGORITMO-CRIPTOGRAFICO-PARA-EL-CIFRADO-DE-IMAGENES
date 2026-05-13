# ══════════════════════════════════════════════════════════════════════════════
#  Makefile — Encrypted Video Streaming (Server + Client)
#
#  Compiles the CUDA encryption core from ../cuda/src/ (WITHOUT modifying it)
#  and the new video streaming server/client applications.
# ══════════════════════════════════════════════════════════════════════════════

NVCC       = nvcc
NVCCFLAGS  = -O3 -rdc=true -arch=sm_89 -ccbin g++-14

# Libraries
LDFLAGS    = $(shell pkg-config --libs opencv4) -lssl -lcrypto -lcudart -lpthread
INCLUDES   = $(shell pkg-config --cflags opencv4) -I../cuda/include -I./common \
             -I/usr/local/cuda/include

# Precision
PRECISION ?= float
ifeq ($(PRECISION),double)
    NVCCFLAGS += -DUSE_DOUBLE_PRECISION
endif

# Directories
CUDA_SRC   = ../cuda/src
BIN_DIR    = ./bin
BUILD_DIR  = ./build

# ── Core source files (from existing project, NOT modified) ──────────────
CORE_CU_SRCS = $(CUDA_SRC)/aux.cu \
               $(CUDA_SRC)/encryption.cu \
               $(CUDA_SRC)/encryption_aux.cu \
               $(CUDA_SRC)/kernels.cu \
               $(CUDA_SRC)/kernels_aux.cu \
               $(CUDA_SRC)/CudaPermutation.cu

CORE_CPP_SRCS = $(CUDA_SRC)/steganography.cpp \
                $(CUDA_SRC)/steganography_reverse.cpp

CORE_CU_OBJS  = $(patsubst $(CUDA_SRC)/%.cu, $(BUILD_DIR)/core_%.o, $(CORE_CU_SRCS))
CORE_CPP_OBJS = $(patsubst $(CUDA_SRC)/%.cpp, $(BUILD_DIR)/core_%.o, $(CORE_CPP_SRCS))
CORE_OBJS     = $(CORE_CU_OBJS) $(CORE_CPP_OBJS)

# pkg-config for libexif (needed by steganography)
LDFLAGS   += $(shell pkg-config --libs libexif 2>/dev/null)
INCLUDES  += $(shell pkg-config --cflags libexif 2>/dev/null)

# ── New streaming source files ───────────────────────────────────────────
COMMON_SRC = ./common/video_crypto.cu
COMMON_OBJ = $(BUILD_DIR)/video_crypto.o

SERVER_SRC = ./server/main_server.cu
SERVER_OBJ = $(BUILD_DIR)/main_server.o

CLIENT_SRC = ./client/main_client.cu
CLIENT_OBJ = $(BUILD_DIR)/main_client.o

# ── Targets ──────────────────────────────────────────────────────────────

.PHONY: all server client clean

all: server client

server: $(BIN_DIR)/video_server
client: $(BIN_DIR)/video_client

# ── Link server ──────────────────────────────────────────────────────────

$(BIN_DIR)/video_server: $(CORE_OBJS) $(COMMON_OBJ) $(SERVER_OBJ)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "✅ Server built: $@"

# ── Link client ──────────────────────────────────────────────────────────

$(BIN_DIR)/video_client: $(CORE_OBJS) $(COMMON_OBJ) $(CLIENT_OBJ)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "✅ Client built: $@"

# ── Compile core .cu files (from original project) ───────────────────────

$(BUILD_DIR)/core_%.o: $(CUDA_SRC)/%.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -dc $< -o $@

# ── Compile core .cpp files (steganography) ──────────────────────────────

$(BUILD_DIR)/core_%.o: $(CUDA_SRC)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -dc $< -o $@

# ── Compile new streaming files ──────────────────────────────────────────

$(BUILD_DIR)/video_crypto.o: $(COMMON_SRC)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -dc $< -o $@

$(BUILD_DIR)/main_server.o: $(SERVER_SRC)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -dc $< -o $@

$(BUILD_DIR)/main_client.o: $(CLIENT_SRC)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -dc $< -o $@

# ── Clean ────────────────────────────────────────────────────────────────

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "🧹 Cleaned."
