from sta_function import result_statistics_1, result_statistics_2, result_statistics_3

path = './regre_result/'
result_statistics_1(path)
result_statistics_2(path)
for scz in ['lishui/', 'ningbo/']:
    path_ = path + scz
    result_statistics_3(path_)
