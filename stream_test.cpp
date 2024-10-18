#include "hnswlib_delete/hnswlib.h"
#include "SIFT_bin_search.h"
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <thread>
#include <vector>
#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>
#include <random>

using namespace std;

template <typename dist_t>
void readData(
    std::ifstream& data_reader,
    dist_t*& data,
    int32_t &dim,
    size_t readLen,
    size_t vectorP
){
    data_reader.read((char*)data+vectorP*dim*sizeof(float), readLen * dim * sizeof(float));
}

int main(int argc, char* argv[]) {
    int32_t dim = 0;               // Dimension of the elements
    int32_t max_elements = 0;   // Maximum number of elements, should be known beforehand
    float* data=nullptr;
    int32_t query_sum=0;
    int32_t query_dim=0;
    float* querys=nullptr;
    int32_t groundtruth_sum=0;
    int32_t groundtruth_dim=0;
    uint32_t* groundtruth=nullptr;

    int circul_sum=100;
    float delete_parts=0.05;
    int delete_model=0;

    string data_file_path="./dataset/sift_learn.fbin";
    string query_file_path="./dataset/sift_query.fbin";
    string groundtruth_file_path="./dataset/sift_query_learn_gt100";
    string search_result_file_path="./search_result";
    string index_prefix="./index/hnsw";
    int K=100;
    vector<size_t>deleteList;
    int M=16;
    int ef=200;
    int num_threads = 64;
    int newLinkSize=M;
    string deleteModelName[]={"VIOLENT_DELETE","PINTOPOUT_DELETE","SEARCH_DELETE","TWOHOP_DELETE","APPROXIMATE_TWOHOP_DELETE","REFACTOR_DELETE"};

    if(argc!=13&&argc!=1){
        cout<<"Missing parameters!!!!!!!!!!!!!!  "<<argc<<endl;
        throw;
    }   
    if(argc>1){
        M=atoi(argv[1]);
        ef=atoi(argv[2]);
        newLinkSize=atoi(argv[3]);
        num_threads=atoi(argv[4]);
        circul_sum=atoi(argv[5]);
        delete_parts=atof(argv[6]);
        delete_model=atoi(argv[7]);
        data_file_path.clear();
        query_file_path.clear();
        index_prefix.clear();
        groundtruth_file_path.clear();
        search_result_file_path.clear();
        data_file_path+=argv[8];
        query_file_path+=argv[9];
        groundtruth_file_path+=argv[10];
        search_result_file_path+=argv[11];
        index_prefix+=argv[12];
    }
    cout<<"-----------------------------------------------------------------------------------"<<endl;
    cout<<"M: "<<M<<" ef: "<<ef<<" newLinkSize: "<<newLinkSize<<" num_threads: "<<num_threads<<"circul_sum: "<<circul_sum<<" delete_parts: "<<delete_parts<<" delete_model: "<<deleteModelName[delete_model]<<endl;
    cout<<"data_file_path: "<<data_file_path<<" query_file_path: "<<query_file_path<<endl<<" groundtruth_file_path: "
        <<groundtruth_file_path<<" search_result_file_path: "<<search_result_file_path<<" index_prefix: "<<index_prefix<<endl;
    cout<<"-----------------------------------------------------------------------------------"<<endl;

    readInitData<float>(dim,max_elements,data,data_file_path);
    readQuerys<float>(query_sum,query_dim,querys,query_file_path);

    cout<<"max_elements: "<<max_elements<<" dim: "<<dim<<endl;
    cout<<"query_sum: "<<query_sum<<" query_dim: "<<query_dim<<endl;
    cout<<"groundtruth_sum: "<<groundtruth_sum<<" groundtruth_dim: "<<groundtruth_dim<<endl;
    // Initing index
    cout<<"Initing index"<<endl;
    hnswlib::L2Space space(dim);

    hnswlib::HierarchicalNSW<float>* alg_hnsw=nullptr;
    pair<float,double> search_result;
    std::ofstream result_writer(search_result_file_path);
    result_writer<<"recall,search_OPS,delete_OPS,insert_OPS"<<endl;
    creat_index(alg_hnsw,index_prefix,&space,M,ef,dim,max_elements,data,num_threads);
    cout<<"start search"<<endl;
    std::ifstream data_reader(data_file_path, std::ios::binary);
    data_reader.seekg(8+max_elements*dim*sizeof(float),ios::beg);
    size_t readLen=1000000;
    size_t vectorP=0;
    size_t fileVectorP=0;
    for(int cir_times=0;cir_times<circul_sum;cir_times++){
        string now_groundTruth_path=groundtruth_file_path+to_string(fileVectorP)+'-'+to_string(fileVectorP+max_elements);
        readGroundTruth<float>(groundtruth_sum,groundtruth_dim,groundtruth,now_groundTruth_path,K);
        for(int i=0;i<groundtruth_sum*groundtruth_dim;i++){
            groundtruth[i]+=fileVectorP;
        }

        search_result=search_index(alg_hnsw,K,query_sum,query_dim,querys,groundtruth_sum,groundtruth_dim,groundtruth,num_threads);
        cout<<"cir_times: "<<cir_times<<'\t'<<"recall: "<<search_result.first<<'\t'<<"search_OPS: "<<search_result.second<<" query\\second\t";
        result_writer<<search_result.first<<','<<search_result.second<<',';

        double deleteTime=deleteIndex(alg_hnsw,fileVectorP,readLen,delete_model,num_threads,newLinkSize);
        cout<<"delete_ops: "<<deleteTime<<" point\\second\t";
        result_writer<<deleteTime<<',';

        data_reader.read((char*)data+vectorP*dim*sizeof(float), readLen * dim * sizeof(float));
        
        double addTime_avg=addPoint(alg_hnsw,fileVectorP+max_elements,readLen,vectorP,num_threads,data,dim);
        cout<<"add_ops: "<<addTime_avg<<" point\\second"<<endl;
        result_writer<<addTime_avg<<endl;

        fileVectorP+=readLen;
        vectorP+=readLen;
        vectorP%=max_elements;
    }
    data_reader.close();
    result_writer.close();
    delete[] data;
    delete[] querys;
    delete[] groundtruth;
    delete alg_hnsw;
    data=nullptr;
    querys=nullptr;
    groundtruth=nullptr;
    alg_hnsw=nullptr;
    return 0;
}