EC_pec.mat\EO_pec.mat\EC_wpli.mat\EO_wpli.mat\EC_icoh.mat\EO_ico.mat由以下数据组成：
overall			1x1 struct			全时段静息态脑电数据进行转换的ROI数据
split1			1x1 struct			前一半时长静息态脑电数据进行转换的ROI数据
split2			1x1 struct			后一半时长静息态脑电数据进行转换的ROI数据
sub_number		129x1 double		ROI数据对应的被试者id号

overall\split1\split2由以下数据组成：
DELTA			129x465 double
THETA			129x465 double
ALPHA			129x465 double
BETA			129x465 double
GAMMA			129x465 double
all				129x465 double



（A）
patient_info.xlsx（以及patient_info.csv）是原始临床信息记录表格，存在以下错误（以下错误已经在patient_info_corr.xlsx（以及patient_info_corr.csv）中进行更正）：
2027 panweipin（名字少一个g，已经修改为panweiping）
3007 chenkai（3007_chenkai_CE20230828_151518_clean.mat少一个下划线，已经修改为3007_chenkai_CE_20230828_151518_clean.mat）
4027 zhuweilin（文件名少一个g，已经修改为zhuweiling）

（B）
原始表格中的列：Unnamed的id号与被试者对应存在错误，patient_info_corr.xlsx（以及patient_info_corr.csv）已经通过被试者名字，在原始.mff文件夹中寻找出对应正确的id号，并在patient_info_corr.xlsx（以及patient_info_corr.csv）表格中添加了正确的id号（列：id）

（C）
patient_info_corr.xlsx（以及patient_info_corr.csv）中共有122名被试者的临床信息，ROI数据中有129名被试者的脑电特征，其中表格缺少以下7名被试者的信息：
1023 fuchangyong
1040 yeweibing
2028 zhangqingfen
2034 zhouhuihui
3009 zhangchangfu
3010 caijunhun
3013 zhangyongming
[1023, 1040, 2028, 2034, 3009, 3010, 3013]

（D）
对于rbans量表，表格缺少以下21+1名被试者的rbans量表数据（其中，id号为2021的被试者比较特殊，只缺少了rbans_3、rbans总分、换算后的记录）：
[1001, 1008, 1009, 1013, 1021, 1025, 2003,
2006, 2012, 2013, 2017, 2039, 3007, 3008,
3014, 3017, 3024, 3025, 3026, 4002, 4014,
2021]