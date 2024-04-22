from main_run import run
from sklearn.linear_model import ElasticNet


SCZ = 'ningbo'  # ['lishui', 'ningbo']
scale = 'panss'  # ['panss', 'rbans']

# ['pec', 'wpli', 'icoh']
feature = ['wpli', ]
# ['merge', 'all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
band = ['merge']
cluster_merge = True  # [True, False]


def model_en(SCZ, scale, feature, band, cluster_merge):
    model_name_select = ['ElasticNet', ]
    model_select = [ElasticNet(random_state=0, alpha=0.15, l1_ratio=0.9)]
    param_grid_select = [{},]

    run(SCZ, scale, feature, band, cluster_merge, model_name_select, model_select, param_grid_select)


model_en(SCZ, scale, feature, band, cluster_merge)
