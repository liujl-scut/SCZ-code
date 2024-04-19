EC_pec.mat\EO_pec.mat\EC_wpli.mat\EO_wpli.mat\EC_icoh.mat\EO_ico.mat由以下数据组成：
overall			1x1 struct			全时段静息态脑电数据进行转换的ROI数据
split1			1x1 struct			前一半时长静息态脑电数据进行转换的ROI数据
split2			1x1 struct			后一半时长静息态脑电数据进行转换的ROI数据
sub_number		96x1 double			ROI数据对应的被试者id号
overall\split1\split2由以下数据组成：
DELTA			96x465 double
THETA			96x465 double
ALPHA			96x465 double
BETA			96x465 double
GAMMA			96x465 double
all				96x465 double

patient_info_corr.xlsx（以及patient_info_corr.csv）中共有83名被试者的临床信息，ROI数据中有96名被试者的脑电特征，其中表格缺少以下13名被试者的信息：
[5, 22, 24, 25, 27, 28, 36, 39, 45, 51, 91, 99, 102]