#include "hnswlib_delete/hnswlib.h"
#include "hnsw_Wolverine/hnsw_Wolverine.h"
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
    readGroundTruth<float>(groundtruth_sum,groundtruth_dim,groundtruth,groundtruth_file_path,K);
    
    // query_sum=100;
    cout<<"max_elements: "<<max_elements<<" dim: "<<dim<<endl;
    cout<<"query_sum: "<<query_sum<<" query_dim: "<<query_dim<<endl;
    cout<<"groundtruth_sum: "<<groundtruth_sum<<" groundtruth_dim: "<<groundtruth_dim<<endl;
    // Initing index
    cout<<"Initing index"<<endl;
    hnswlib::L2Space space(dim);

    hnswlib::HierarchicalNSW<float>* alg_hnsw=nullptr;
    hnswlib::HierarchicalNSW<float>* alg_hnsw_old=nullptr;
    pair<float,double> search_result;
    std::ofstream result_writer(search_result_file_path);
    result_writer<<"usedTimes,amount,part"<<endl;
    unsigned int seed=100;
    default_random_engine random(seed);
    uniform_int_distribution<size_t> dis1(0, max_elements-1);
    creat_index(alg_hnsw,index_prefix,&space,M,ef,dim,max_elements,data,num_threads);
    creat_index(alg_hnsw_old,index_prefix,&space,M,ef,dim,max_elements,data,num_threads);
    cout<<"maxlevel_: "<<alg_hnsw->maxlevel_<<endl;

    cout<<"start delete"<<endl;
    creat_deleteList(deleteList,max_elements*delete_parts*2,max_elements*delete_parts,random,dis1);
    double deleteTime=deleteIndex(alg_hnsw,deleteList,delete_model,num_threads,newLinkSize);
    vector<pair<pair<tableint,tableint>,size_t>> newEdgeList;
    
    newEdgeList=getNewEdge(alg_hnsw_old,alg_hnsw);
    cout<<"getNewEdge finish!!!!!!!"<<endl;

    cout<<"start search"<<endl;
    vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>> result(query_sum);
    for(size_t row=0;row<query_sum;row++){
        vector<tableint> visitedNodes;
        result[row] = alg_hnsw->searchKnn((void*)((char*)querys + row * query_dim * sizeof(float)), K,visitedNodes);
        ParallelFor(0, newEdgeList.size(), num_threads, [&](size_t row, size_t threadId) {
            for(int i=0;i<visitedNodes.size();i++){
                if(visitedNodes[i]==newEdgeList[row].first.first){
                    tableint *data = alg_hnsw->get_linklist_at_level(visitedNodes[i], 0);
                    int size = alg_hnsw->getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
                    if(find(datal,datal+size,newEdgeList[row].first.second)!=datal+size&&find(visitedNodes.begin()+i,visitedNodes.end(),newEdgeList[row].first.second)!=visitedNodes.end()){
                        newEdgeList[row].second++;
                    }
                    continue;
                }
            }
        });
        show_progress_bar(row,query_sum);
    }
    cout<<"finish!!!!!!!"<<endl;

    vector<int> usedTimes(query_sum+1,0);
    for(int i=0;i<newEdgeList.size();i++){
        usedTimes[newEdgeList[i].second]++;
        // result_writer<<'('<<newEdgeList[i].first.first<<','<<newEdgeList[i].first.second<<"): "<<newEdgeList[i].second<<endl;
    }
    int partSum=0;
    for(int i=0;i<usedTimes.size();i++){
        partSum+=usedTimes[i];
        if(i%(usedTimes.size()/100)==0){
            result_writer<<i<<","<<partSum<<","<<(float)partSum/(float)newEdgeList.size()<<endl;
            partSum=0;
        }
    }

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