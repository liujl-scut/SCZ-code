from main_run import run
from sklearn.kernel_ridge import KernelRidge


SCZ = 'ningbo'  # ['lishui', 'ningbo']
scale = 'panss'  # ['panss', 'rbans']

# ['pec', 'wpli', 'icoh']
feature = ['pec', ]
# ['merge', 'all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
band = ['ALPHA', 'BETA']
cluster_merge = False  # [True, False]


def model_kr(SCZ, scale, feature, band, cluster_merge):
    model_name_select = ['KernelRidge',]
    model_select = [KernelRidge(),]
    param_grid_select = [{'learning_rate_init': [0.001, 0.01, 0.1, 0.2]},]

    run(SCZ, scale, feature, band, cluster_merge, model_name_select, model_select, param_grid_select)


model_kr(SCZ, scale, feature, band, cluster_merge)
