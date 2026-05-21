# Variables
NVCC = nvcc

# Normal flags
NVCCFLAGS_NORMAL = -O3 -rdc=true

# Debug fags
NVCCFLAGS_DEBUG = -G -g -O0 -rdc=true

# Libraries and includes
LDFLAGS = $(shell pkg-config --libs opencv4) $(shell pkg-config --libs libexif) -lssl -lcrypto -lcudart -Xlinker /usr/lib/x86_64-linux-gnu/libstdc++.so.6 
CXXINCLUDES = $(shell pkg-config --cflags opencv4) $(shell pkg-config --cflags libexif) -I./cuda/include -I/usr/local/cuda/include

# Directories
SRC_DIR = ./cuda/src
BIN_DIR = ./cuda/bin
INCLUDE_DIR = ./cuda/include

# Files
SRCS_CU = $(wildcard $(SRC_DIR)/*.cu)
SRCS_CPP = $(wildcard $(SRC_DIR)/*.cpp)
OBJS_CU = $(SRCS_CU:.cu=.o)
OBJS_CPP = $(SRCS_CPP:.cpp=.o)
OBJS = $(OBJS_CU) $(OBJS_CPP)
TARGET = $(BIN_DIR)/cipher.out

# Mode selection: normal o debug
MODE ?= normal
ifeq ($(MODE),debug)
    CXXFLAGS = $(CXXFLAGS_DEBUG)
    NVCCFLAGS = $(NVCCFLAGS_DEBUG)
else
    NVCCFLAGS = $(NVCCFLAGS_NORMAL)
endif

# Precission selection (new)
PRECISION ?= float
ifeq ($(PRECISION),double)
    NVCCFLAGS += -DUSE_DOUBLE_PRECISION
endif

 NVCCFLAGS += -ccbin g++-12

# main rule
$(TARGET):  $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@ $(LDFLAGS)

# .cu compilation
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(CXXINCLUDES) -dc $< -o $@

# .cpp compilation (C++ files like steganography)
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) $(CXXINCLUDES) -dc $< -o $@

# clean
clean:
	rm -rf $(BIN_DIR) $(SRC_DIR)/*.o

clean_objs:
	rm -f $(SRC_DIR)/*.o
