# Variables
CXX = g++
NVCC = nvcc

# Flags normales
CXXFLAGS_NORMAL = -std=c++17 -O2 -Wall -pedantic
NVCCFLAGS_NORMAL = -O2 -rdc=true

# Flags de depuración
CXXFLAGS_DEBUG = -std=c++17 -g -O0 -Wall -pedantic
NVCCFLAGS_DEBUG = -G -g -O0 -rdc=true

# Librerías e includes
LDFLAGS = $(shell pkg-config --libs opencv4) -lssl -lcrypto
CXXINCLUDES = $(shell pkg-config --cflags opencv4) -I./include

# Directorios
SRC_DIR = ./src
BIN_DIR = ./bin
INCLUDE_DIR = ./include

# Archivos
SRCS_CPP = $(wildcard $(SRC_DIR)/*.cpp)
SRCS_CU = $(wildcard $(SRC_DIR)/*.cu)
OBJS_CPP = $(SRCS_CPP:.cpp=.o)
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

# Regla principal
$(TARGET): $(OBJS_CPP) $(OBJS_CU)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(OBJS_CPP) $(OBJS_CU) -o $@ $(LDFLAGS)

# Compilación de .cpp
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c $< -o $@

# Compilación de .cu
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(CXXINCLUDES) -c $< -o $@

# Limpieza
clean:
	rm -rf $(BIN_DIR) $(SRC_DIR)/*.o

clean_objs:
	rm -f $(SRC_DIR)/*.o
