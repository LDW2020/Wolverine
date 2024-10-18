CXX=g++
GRAPH=HNSW
DELETE_MODEL=DM_ATWOHOP
CXXFLAGS= -I. -std=c++17 -Ofast -pthread -fopenmp -D__AVX__ -mavx2 -D$(GRAPH) #-D$(DELETE_MODEL)# -DDATATYPE=float  
TARGET=SIFT_bin_search
SRC=SIFT_bin_search.cpp


all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)