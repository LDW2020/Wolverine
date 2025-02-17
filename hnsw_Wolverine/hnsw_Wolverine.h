#include "hnswlib.h"
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
#include <omp.h>

using namespace std;
typedef unsigned int tableint;

// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}
template <typename dist_t>
void readInitData(int32_t &dim,int32_t &max_elements,dist_t*& data,string data_file_path){
    // read data
    cout<<"Read data"<<endl;
    std::ifstream data_reader(data_file_path, std::ios::binary);
    if(!data_reader.is_open()){
        cout<<"data_reader open failed!!!!"<<endl;
        throw;
    }
    data_reader.read((char*)&max_elements,sizeof(int32_t));
    data_reader.read((char*)&dim,sizeof(int32_t));
    if(max_elements>10000000){
        cout<<"max_elements>10000000"<<endl;
        max_elements=10000000;
    }
    data = new dist_t[dim * max_elements];
    data_reader.read((char*)data, max_elements * dim * sizeof(dist_t));
    data_reader.close();
}

template <typename dist_t>
void readQuerys(int32_t &query_sum,int32_t &query_dim,dist_t*& querys,string query_file_path){
    // Read query
    cout<<"Read query"<<endl;
    std::ifstream query_reader(query_file_path, std::ios::binary);
    if(!query_reader.is_open()){
        cout<<"query_reader open failed!!!!"<<endl;
        throw;
    }
    query_reader.read((char*)&query_sum,sizeof(int32_t));
    query_reader.read((char*)&query_dim,sizeof(int32_t));
    querys = new dist_t[query_dim * query_sum];
    query_reader.read((char*)querys, query_sum * query_dim * sizeof(dist_t));
    query_reader.close();
}

template <typename dist_t>
void readGroundTruth(int32_t &groundtruth_sum,int32_t &groundtruth_dim,uint32_t*& groundtruth,string groundtruth_file_path,int K){
    // Read groundtruth
    // cout<<"Read groundtruth"<<endl;
    std::ifstream groundtruth_reader(groundtruth_file_path, std::ios::binary);
    if(!groundtruth_reader.is_open()){
        cout<<"groundtruth_reader open failed!!!!"<<endl;
        throw;
    }
    groundtruth_reader.read((char*)&groundtruth_sum,sizeof(int32_t));
    groundtruth_reader.read((char*)&groundtruth_dim,sizeof(int32_t));
    if(groundtruth==nullptr){
        groundtruth = new uint32_t[groundtruth_dim * groundtruth_sum];
    }
    groundtruth_reader.read((char*)groundtruth, groundtruth_sum * groundtruth_dim * sizeof(uint32_t));
    groundtruth_reader.close();
    if(groundtruth_dim<K){
        cout<<"groundtruth_dim too small!!!!"<<endl;
        throw;
    }
}

template <typename dist_t>
void creat_index(hnswlib::HierarchicalNSW<dist_t>*& alg_hnsw,string index_prefix,hnswlib::L2Space *space,int M,int ef,int32_t dim,int32_t max_elements,dist_t* data,int num_threads){
    #ifdef LAYER0
    index_prefix+="_R0";
    #endif

    #ifdef NSW
    string index_path=index_prefix+".nswindex";
    #elif defined TestVamana
    string index_path=index_prefix+".vamanaindex";
    #elif defined NSG
    string index_path=index_prefix+".hnswnsgindex";
    #else
    string index_path=index_prefix+".hnswindex";
    #endif
    
    if(access(index_path.c_str(),F_OK)==0){
        cout<<"read index: "<<index_path<<endl;
        alg_hnsw = new hnswlib::HierarchicalNSW<dist_t>(space, index_path,false,0);
    }
    else{
        cout<<"create index: "<<index_path<<endl;
        alg_hnsw = new hnswlib::HierarchicalNSW<dist_t>(space, max_elements+1, M, ef,100);
        // #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 8192)
        // for(size_t row=0;row<max_elements;row++){
        //     alg_hnsw->addPoint((void*)(data + dim * row), row);
        // }
        ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
            alg_hnsw->addPoint((void*)(data + dim * row), row);
        });
        alg_hnsw->saveIndex(index_path);
    }
    // alg_hnsw->setInLinkList();
    // cout<<"inEquOut rate: "<<alg_hnsw->inlinkequoutlink()<<endl;
}

template <typename dist_t>
pair<float,double> search_index(hnswlib::HierarchicalNSW<dist_t>* alg_hnsw,int K,
    int32_t query_sum,int32_t query_dim,dist_t* querys,
    int32_t groundtruth_sum,int32_t groundtruth_dim,uint32_t* groundtruth,int num_threads){
    float correct_sum = 0;
    float* corrects = new float[query_sum];
    memset(corrects,0,query_sum*sizeof(float));
    struct timeval search_start_time,search_end_time;
    double search_time_sum = 0;
    double* search_times = new double[query_sum];
    memset(search_times,0,query_sum*sizeof(double));
    vector<std::priority_queue<std::pair<dist_t, hnswlib::labeltype>>> result(query_sum);
    gettimeofday(&search_start_time,NULL);
    // #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
    // for(size_t row=0;row<query_sum;row++){
    //     result[row] = alg_hnsw->searchKnn((void*)((char*)querys + row * query_dim * sizeof(dist_t)), K);
    // }
    ParallelFor(0, query_sum, num_threads, [&](size_t row, size_t threadId) {
        result[row] = alg_hnsw->searchKnn((void*)((char*)querys + row * query_dim * sizeof(dist_t)), K);
    });
    gettimeofday(&search_end_time,NULL);
    search_time_sum=(double)(search_end_time.tv_sec-search_start_time.tv_sec)+(double)(search_end_time.tv_usec-search_start_time.tv_usec)/1000000;
    ParallelFor(0, query_sum, num_threads, [&](size_t row, size_t threadId) {
        int groundtruth_search_startp=row*groundtruth_dim;
        int groundtruth_search_endp=groundtruth_search_startp+K;
        while (!result[row].empty()){
            hnswlib::labeltype label = result[row].top().second;
            uint32_t* res=find(groundtruth+groundtruth_search_startp,groundtruth+groundtruth_search_endp,label);
            if(res!=groundtruth+groundtruth_search_endp){
                corrects[row]+=1;
            }
            result[row].pop();
        }
    });
    for(int i=0;i<query_sum;i++){
        correct_sum+=corrects[i];
    }
    float recall = correct_sum / query_sum / K;
    return make_pair(recall,query_sum/search_time_sum);
}

void creat_deleteList(vector<size_t>&deleteList,int pre_creat_sum,int list_len,default_random_engine& random,uniform_int_distribution<size_t>& dis1){
    deleteList.clear();
    vector<size_t> deleteList_temp;
    while(deleteList.size()<list_len){
        for(int i=0;i<pre_creat_sum;i++){
            deleteList_temp.emplace_back((size_t)(dis1(random)));
        }
        sort(deleteList_temp.begin(),deleteList_temp.end());
        for(int i=0;i<deleteList_temp.size()-1;i++){
            if(deleteList_temp[i]!=deleteList_temp[i+1]){
                deleteList.emplace_back(deleteList_temp[i]);
                if(deleteList.size()>=list_len){
                    break;
                }
            }
        }
        pre_creat_sum*=2;
    }
    random_shuffle(deleteList.begin(),deleteList.end());
}

void write_Vector(vector<size_t> vec,string file_path){
    std::ofstream vec_writer(file_path);
    for(int i=0;i<vec.size();i++){
        vec_writer<<vec[i];
        if(i!=vec.size()-1)
            vec_writer<<"\n";
    }
    vec_writer.close();
}

#define mulThreadsDelete

template <typename dist_t>
double psedo_deleteIndex(hnswlib::HierarchicalNSW<dist_t>* alg_hnsw,vector<size_t> deleteList,int delete_model,int num_threads,int newLinkSize){
    struct timeval delete_start_time,delete_end_time;
    double delete_time=0;
    gettimeofday(&delete_start_time,NULL);
    for(int row=0;row<deleteList.size();row++){
        alg_hnsw->markDelete(deleteList[row]);
    }
    gettimeofday(&delete_end_time,NULL);
    delete_time=(double)(delete_end_time.tv_sec-delete_start_time.tv_sec)+(double)(delete_end_time.tv_usec-delete_start_time.tv_usec)/1000000;
    return deleteList.size()/delete_time;
}

template <typename dist_t>
double deleteIndex(hnswlib::HierarchicalNSW<dist_t>* alg_hnsw,vector<size_t> deleteList,int delete_model,int num_threads,int newLinkSize){
    struct timeval delete_start_time,delete_end_time;
    double delete_time=0;
    #ifdef mulThreadsDelete
    gettimeofday(&delete_start_time,NULL);
    alg_hnsw->patchDelete(deleteList,delete_model,newLinkSize,num_threads);
    gettimeofday(&delete_end_time,NULL);
    delete_time=(double)(delete_end_time.tv_sec-delete_start_time.tv_sec)+(double)(delete_end_time.tv_usec-delete_start_time.tv_usec)/1000000;
    #else
    for(int row=0;row<deleteList.size();row++){
        gettimeofday(&delete_start_time,NULL);
        alg_hnsw->deletePoint(deleteList[row],delete_model);
        gettimeofday(&delete_end_time,NULL);
        delete_time+=(double)(delete_end_time.tv_sec-delete_start_time.tv_sec)+(double)(delete_end_time.tv_usec-delete_start_time.tv_usec)/1000000;
    }
    #endif
    return deleteList.size()/delete_time;
}

template <typename dist_t>
double deleteIndex(hnswlib::HierarchicalNSW<dist_t>* alg_hnsw,size_t deleteStart,size_t deleteLen,int delete_model,int num_threads,int newLinkSize){
    struct timeval delete_start_time,delete_end_time;
    double delete_time=0;
    gettimeofday(&delete_start_time,NULL);
    alg_hnsw->patchDelete(deleteStart,deleteLen,delete_model,newLinkSize,num_threads);
    gettimeofday(&delete_end_time,NULL);
    delete_time+=(double)(delete_end_time.tv_sec-delete_start_time.tv_sec)+(double)(delete_end_time.tv_usec-delete_start_time.tv_usec)/1000000;
    return deleteLen/delete_time;
}

template <typename dist_t>
double addPoint(hnswlib::HierarchicalNSW<dist_t>* alg_hnsw,vector<size_t> deleteList,int num_threads,dist_t* data,int32_t dim){
    double addTime_avg=0;
    vector<double> addTimes(deleteList.size(),0);
    struct timeval add_start_time,add_end_time;
    gettimeofday(&add_start_time,NULL);
    // #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
    // for(size_t row=0;row<deleteList.size();row++){
    //     alg_hnsw->addPoint((void*)((char*)data + dim * deleteList[row] * sizeof(dist_t)), deleteList[row]);
    // }
    ParallelFor(0, deleteList.size(), num_threads, [&](size_t row, size_t threadId) {
        alg_hnsw->addPoint((void*)((char*)data + dim * deleteList[row] * sizeof(dist_t)), deleteList[row]);
    });
    gettimeofday(&add_end_time,NULL);
    addTime_avg+=(double)(add_end_time.tv_sec-add_start_time.tv_sec)+(double)(add_end_time.tv_usec-add_start_time.tv_usec)/1000000;
    return deleteList.size()/addTime_avg;
}

template <typename dist_t>
double addPoint(hnswlib::HierarchicalNSW<dist_t>* alg_hnsw,size_t addStart,size_t addLen,size_t data_offset,int num_threads,dist_t* data,int32_t dim){
    double addTime_avg=0;
    vector<double> addTimes(addLen,0);
    struct timeval add_start_time,add_end_time;
    gettimeofday(&add_start_time,NULL);
    ParallelFor(0,addLen, num_threads, [&](size_t row, size_t threadId) {
        alg_hnsw->addPoint((void*)((char*)data + dim * (data_offset + row) * sizeof(dist_t)), addStart + row);
    });
    gettimeofday(&add_end_time,NULL);
    addTime_avg+=(double)(add_end_time.tv_sec-add_start_time.tv_sec)+(double)(add_end_time.tv_usec-add_start_time.tv_usec)/1000000;
    return addLen/addTime_avg;
}


template <typename dist_t>
vector<pair<pair<tableint,tableint>,size_t>> getNewEdge(hnswlib::HierarchicalNSW<dist_t>* alg_hnsw1,hnswlib::HierarchicalNSW<dist_t>* alg_hnsw2){
    vector<pair<pair<tableint,tableint>,size_t>> newEdgeList;
    for(tableint i=0;i<alg_hnsw2->cur_element_count;i++){
        if(alg_hnsw2->deleteFlags[i]==false){
            tableint *data2 = alg_hnsw2->get_linklist_at_level(i, 0);
            int size2 = alg_hnsw2->getListCount(data2);
            tableint *datal2 = (tableint *) (data2 + 1);

            tableint *data1 = alg_hnsw1->get_linklist_at_level(i, 0);
            int size1 = alg_hnsw1->getListCount(data1);
            tableint *datal1 = (tableint *) (data1 + 1);

            for(size_t j=0;j<size2;j++){
                if(find(datal1,datal1+size1,datal2[j])==datal1+size1){
                    newEdgeList.emplace_back(make_pair(make_pair(i,datal2[j]),0));
                }
            }
        }
    }
    return newEdgeList;
}

void show_progress_bar(int progress, int total) {
    const int bar_width = 100;  // 进度条的宽度
    float progress_percentage = (float)progress / total;  // 计算进度百分比
    int pos = bar_width * progress_percentage;  // 计算已完成的进度部分

    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos)
            std::cout << "#";  // 显示进度
        else
            std::cout << " ";  // 显示剩余部分
    }

    std::cout << "] " << int(progress_percentage * 100.0) << "%";
    std::flush(std::cout);  // 强制刷新输出，确保进度条及时更新
}