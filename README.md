# Wolverine
This is the source code of Wolverine.

Our paper, "Wolverine: Highly Efficient Monotonic Search Path Repair for Graph-based ANN Index Updates", is submitted to SIGMOD2025.

## Directory description
  * data: The dataset required for Wolverine has the following format: the first row indicates the length of the time series, the first column contains the data points, and the second column is the labels (where 0 represents normal and 1 represents an anomaly).
  * hnsw_Wolverine: The source code for HNSW+Wolverine.

## Linux build
All experiments are conducted on a machine with an AMD EPYC 7K62 48-Core Processor CPU and 256GB of memory.

### Prerequisites
 * Ubuntu 22.04.02 LTS
 * GCC 8.3.1
 * BLAS
   
   `$ sudo apt-get update`
   
   `$ sudo apt-get install libblas-dev liblapack-dev`

### Build and run steps
 * Download this repository and change to the Wolverine folder.

   `$ cd Wolverine`
 * Create a "build" directory inside it.

   `$ mkdir build`
 * Change to the "build" directory.

   `$ cd build`
 * Run.

   `$ cmake ..`
   
   `$ make`
   
   `$ ./main` or `$ cd ..` then `$ bash run.sh`
