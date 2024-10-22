declare -A deletemodelmap;
deletemodelmap['Do']=0
deletemodelmap['DwFC']=1
deletemodelmap['Wolverine']=2
deletemodelmap['WolverinePro']=3
deletemodelmap['WolverineProMax']=4

dataset_type='sift_1M' #
M=32
ef=300
candLimit=32
thr=64
circul_sum=50
delete_rate=0.01
delete_model='WolverineProMax'

dataset_path='./datasets/sift_learn.fbin'
query_set_path='./datasets/sift_query.fbin'
groundtruth_path='./datasets/sift_query_learn_gt100'
index_path_path='./index/'${dataset_type}'_M'${M}'_ef'${ef}
search_result_path='./result/TESThnsw_'${dataset_type}'_'${delete_model}'_M'${M}'_ef'${ef}'_thr'${thr}'_Drate'${delete_rate}'_candLimit'${candLimit}'.csv'

make GRAPH=HNSW clean
make GRAPH=HNSW
./hnsw_Wolverine_test ${M} ${ef} ${candLimit} ${thr} ${circul_sum} ${delete_rate} ${deletemodelmap[$delete_model]} ${dataset_path} ${query_set_path} ${groundtruth_path} ${search_result_path} ${index_path_path}



