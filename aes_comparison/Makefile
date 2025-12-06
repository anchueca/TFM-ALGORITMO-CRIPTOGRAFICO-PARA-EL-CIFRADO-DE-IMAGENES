CXX = g++
CXXFLAGS = -O3 -Wall
LDFLAGS = -lcrypto

TARGET = aes_tool
SRC = aes_tool.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
