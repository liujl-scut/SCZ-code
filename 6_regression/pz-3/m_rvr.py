from main_run import run
from skrvm import RVR


SCZ = 'lishui'  # ['lishui', 'ningbo']
scale = 'panss'  # ['panss', 'rbans']

# ['pec', 'wpli', 'icoh']
feature = ['wpli', ]
# ['merge', 'all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
band = ['all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
cluster_merge = False  # [True, False]


def model_rvr(SCZ, scale, feature, band, cluster_merge):
    model_name_select = ['RVR',]
    model_select = [RVR(kernel='linear', n_iter=10000),]
    param_grid_select = [{},]

    run(SCZ, scale, feature, band, cluster_merge, model_name_select, model_select, param_grid_select)


model_rvr(SCZ, scale, feature, band, cluster_merge)
