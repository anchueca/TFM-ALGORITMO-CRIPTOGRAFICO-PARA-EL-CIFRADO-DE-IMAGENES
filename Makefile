# Variables
NVCC = nvcc

# Flags normales
NVCCFLAGS_NORMAL = -O2 -rdc=true

# Flags de depuración
NVCCFLAGS_DEBUG = -G -g -O0 -rdc=true

# Librerías e includes
LDFLAGS = $(shell pkg-config --libs opencv4) -lssl -lcrypto -lcudart 
CXXINCLUDES = $(shell pkg-config --cflags opencv4) -I./include -I/usr/local/cuda/include

# Directorios
SRC_DIR = ./cuda/src
BIN_DIR = ./cuda/bin
INCLUDE_DIR = ./cuda/include

# Archivos
SRCS_CU = $(wildcard $(SRC_DIR)/*.cu)
OBJS_CU = $(SRCS_CU:.cu=.o)
TARGET = $(BIN_DIR)/cipher.out

# Selección de modo: normal o debug
MODE ?= normal
ifeq ($(MODE),debug)
    CXXFLAGS = $(CXXFLAGS_DEBUG)
    NVCCFLAGS = $(NVCCFLAGS_DEBUG)
else
    CXXFLAGS = $(CXXFLAGS_NORMAL)
    NVCCFLAGS = $(NVCCFLAGS_NORMAL)
endif

NVCCFLAGS += -ccbin g++-12

# Regla principal
$(TARGET):  $(OBJS_CU)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(OBJS_CU) -o $@ $(LDFLAGS)

# Compilación de .cu
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(CXXINCLUDES) -dc $< -o $@

# Limpieza
clean:
	rm -rf $(BIN_DIR) $(SRC_DIR)/*.o

clean_objs:
	rm -f $(SRC_DIR)/*.o
