# Wolverine
This is the source code of Wolverine.

Our paper, "Wolverine: Highly Efficient Monotonic Search Path Repair for Graph-based ANN Index Updates", is submitted to SIGMOD2025.

## Directory description
  * datasets: The dataset required for Wolverine has the following format: the first 4 bytes (int32_t) indicates the size of the datasets, the second 4 bytes (int32_t) indicates the dimensions of the vectors, then each vector data (float) is closely packed.
  * hnsw_Wolverine: The source code for HNSW+Wolverine.

## Linux build
All experiments are conducted on a machine with an AMD EPYC 7K62 48-Core Processor CPU and 256GB of memory.

### Prerequisites
 * Ubuntu 20.04.6 LTS
 * GCC 9.4.0

### Build and run steps
 * Download this repository and change to the Wolverine folder.

   `$ cd Wolverine`
   
   `$ mkdir index`

   `$ ./test.sh`
