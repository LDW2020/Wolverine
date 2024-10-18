declare -A deletemodelmap;
deletemodelmap['VIOLENT_DELETE']=0
deletemodelmap['PINTOPOUT_DELETE']=1
deletemodelmap['SEARCH_DELETE']=2
deletemodelmap['TWOHOP_DELETE']=3
deletemodelmap['APPROXIMATE_TWOHOP_DELETE']=4

declare -A datamap;
datamap['sift_learn']='/data/dataset/SIFT/sift/fbin/sift_learn.fbin'
datamap['sift_1M']='/data/dataset/SIFT/sift/fbin/sift_base.fbin'
datamap['sift_10M']='/data/dataset/SIFT/bigann/fbin/bigann_learn_10M_random.fbin'
datamap['gist_1M']='/data/dataset/GIST/fbin/gist_base.fbin'
datamap['sift_100M']='/data/dataset/SIFT/bigann/fbin/bigann_learn.fbin'

declare -A querymap;
querymap['sift_learn']='/data/dataset/SIFT/sift/fbin/sift_query.fbin'
querymap['sift_1M']='/data/dataset/SIFT/sift/fbin/sift_query.fbin'
querymap['sift_10M']='/data/dataset/SIFT/bigann/fbin/bigann_query.fbin'
querymap['gist_1M']='/data/dataset/GIST/fbin/gist_query.fbin'
querymap['sift_100M']='/data/dataset/SIFT/bigann/fbin/bigann_query.fbin'

declare -A groundtruthmap;
groundtruthmap['sift_learn']='/data/dataset/SIFT/sift/gnd/sift_query_learn_gt100'
groundtruthmap['sift_1M']='/data/dataset/SIFT/sift/gnd/sift_query_base_gt100'
groundtruthmap['sift_10M']='/data/dataset/SIFT/bigann/fgnd/bigann_query_learn_10M_random_gt100'
groundtruthmap['gist_1M']='/data/dataset/GIST/fgnd/gist_query_base_gt100'
groundtruthmap['sift_100M']='/data/dataset/SIFT/bigann/fgnd/bigann_query_learn_gt100'

dataset_type='sift_1M' #
M=64
ef=100
candLimit=64
thr=64
circul_sum=50
delete_rate=0.05
delete_model='APPROXIMATE_TWOHOP_DELETE'

dataset_path=${datamap[$dataset_type]}
query_set_path=${querymap[$dataset_type]}
groundtruth_path=${groundtruthmap[$dataset_type]}
index_path_path='/data/GraphUpdate/index/'${dataset_type}'_M'${M}'_ef'${ef}
search_result_path='./result/TESThnsw_'${dataset_type}'_'${delete_model}'_M'${M}'_ef'${ef}'_thr'${thr}'_Drate'${delete_rate}'_candLimit'${candLimit}'.csv'

make GRAPH=VAMANA clean
make GRAPH=VAMANA
./SIFT_bin_search ${M} ${ef} ${candLimit} ${thr} ${circul_sum} ${delete_rate} ${deletemodelmap[$delete_model]} ${dataset_path} ${query_set_path} ${groundtruth_path} ${search_result_path} ${index_path_path}

# make -f Makefile_NSW clean
# make -f Makefile_NSW
# ./SIFT_bin_search ${M} ${ef} ${candLimit} ${thr} ${circul_sum} ${delete_rate} ${deletemodelmap[$delete_model]} ${dataset_path} ${query_set_path} ${groundtruth_path} ${search_result_path} ${index_path_path}


# ef=100
# index_path_path='/data/GraphUpdate/index/'${dataset_type}'_M'${M}'_ef'${ef}
# make -f Makefile_Vamana clean
# make -f Makefile_Vamana
# ./SIFT_bin_search ${M} ${ef} ${candLimit} ${thr} ${circul_sum} ${delete_rate} ${deletemodelmap[$delete_model]} ${dataset_path} ${query_set_path} ${groundtruth_path} ${search_result_path} ${index_path_path}




