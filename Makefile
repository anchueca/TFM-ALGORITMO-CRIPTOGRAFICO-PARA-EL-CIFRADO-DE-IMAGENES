# Variables
NVCC = nvcc

# Normal flags
NVCCFLAGS_NORMAL = -O3 -rdc=true #-use_fast_math

# Debug fags
NVCCFLAGS_lDEBUG = -G -g -O0 -rdc=true

# Libraries and includes
LDFLAGS = $(shell pkg-config --libs opencv4) -lssl -lcrypto -lcudart 
CXXINCLUDES = $(shell pkg-config --cflags opencv4) -I./cuda/include -I/usr/local/cuda/include

# Directories
SRC_DIR = ./cuda/src
BIN_DIR = ./cuda/bin
INCLUDE_DIR = ./cuda/include

# Files
SRCS_CU = $(wildcard $(SRC_DIR)/*.cu)
OBJS_CU = $(SRCS_CU:.cu=.o)
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
$(TARGET):  $(OBJS_CU)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(OBJS_CU) -o $@ $(LDFLAGS)

# .cu compilation
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(CXXINCLUDES) -dc $< -o $@

# clean
clean:
	rm -rf $(BIN_DIR) $(SRC_DIR)/*.o

clean_objs:
	rm -f $(SRC_DIR)/*.o
