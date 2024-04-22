from main_run import run
from sklearn.svm import SVR


SCZ = 'lishui'  # ['lishui', 'ningbo']
scale = 'panss'  # ['panss', 'rbans']

# ['pec', 'wpli', 'icoh']
feature = ['wpli', ]
# ['merge', 'all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
band = ['all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
cluster_merge = False  # [True, False]


def model_svr(SCZ, scale, feature, band, cluster_merge):
    model_name_select = ['SVR', ]
    model_select = [SVR(kernel='linear')]
    param_grid_select = [{'C': [0.0001, 0.001, 0.01, 0.1]}, ]

    run(SCZ, scale, feature, band, cluster_merge, model_name_select, model_select, param_grid_select)


model_svr(SCZ, scale, feature, band, cluster_merge)
