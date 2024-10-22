CXX=g++
GRAPH=HNSW
DELETE_MODEL=DM_ATWOHOP
CXXFLAGS= -I. -std=c++17 -Ofast -pthread -fopenmp -D__AVX__ -mavx2 -D$(GRAPH)
TARGET=hnsw_Wolverine_test
SRC=hnsw_Wolverine_test.cpp


all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)