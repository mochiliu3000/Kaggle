#config
from param_config import config
#data process, data input, data output...
from core.data_proc import Data_proc
#feature extract
from core.feat_extr import Feat_extr
#single models
from core.models.hyperopt_tune import model_hyperopt_tune
#ensemble selection
from core.ensemble import ensemble_hyperopt_tune

#from core.model_sgle import Model_sgle
#import core.test as test
#(precise, cvfold...)


##################
## 1.Input Data ##
##################
dfTrainX_origin, dfTrainY_origin, dfTestX_origin = Data_proc.input(config, header=0)
print("1.Data Inputted from " + config.train_path + " and " + config.test_path)
print("========================================================")


########################
## 2.Pre-process Data ##
########################
# 2.1 Categorize
# Ticket
# TODO: Using column names is better 
categorize_labels = (7,)
categorize_methods = ("has_char",)
dfTrainX, dfTestX = Data_proc.categorize(dfTrainX_origin, dfTestX_origin, categorize_labels, categorize_methods)
# 2.2 NA_Managment
# Age, Fare, Embarked
na_labels = (4,8,10)
na_methods = ("fill_median", "fill_median", "fill_mode")
dfTrainX, dfTestX = Data_proc.na_manage(dfTrainX, dfTestX, na_labels, na_methods)
# 2.3 Outlier
# 2.4 Scale
#scale_labels = range(dfTrainX.shape[1])
#scale_methods = ("__normalize",) * 4
#dfTrainX, dfTestX = Data_proc.scale(dfTrainX_origin, dfTestX_origin, scale_labels, scale_methods)
# 2.5 PrepCV and Persist
#TODO: stratifiedKFold warning!!!
skf = Data_proc.prepCV(config, header=0)
Data_proc.persist(config, (skf,), ("stratifiedKFold_ind",))
print("2.Pre-processing Finished")
print("========================================================")


##########################
## 3.Feature Extraction ##
##########################
# 3.0 Fetch ID
dfTestId = dfTestX.ix[:,0]
# 3.1 Add
# 3.2 Drop
# PassengerId, Name, Cabin
dfTrainX, dfTestX = Feat_extr.drop(dfTrainX, dfTestX, (0,9))
# 3.2 Transform
# Sex, Embarked
transform_X_labels = (1,2,8)
transform_X_methods = (("extract", "map_with_dict", "to_value"), ("to_value",), ("to_value",))
dfTrainX, dfTestX = Feat_extr.transform(dfTrain=dfTrainX, dfTest=dfTestX, methods=transform_X_methods, labels=transform_X_labels)
transform_Y_method = (("to_value",),)
dfTrainY = Feat_extr.transform(dfTrain=dfTrainY_origin, methods=transform_Y_method)
dfTrainY = dfTrainY[0]

# 3.3 Combine
combine_X_methods = (("comb_add",),)
combine_X_labels = ((4,5),)
dfTrainX, dfTestX = Feat_extr.combine(dfTrain=dfTrainX, dfTest=dfTestX, methods=combine_X_methods, labels=combine_X_labels)
# 3.4 Persist
print("--------------------------------------------")
print(dfTrainX.ix[:5,])
#http://stackoverflow.com/questions/29530232/python-pandas-check-if-any-value-is-nan-in-dataframe
print("Number NA in %s: %d" % ("dfTrainX", dfTrainX.isnull().sum().sum()))
print("If NA exists in %s" % "dfTrainX")
print(dfTrainX.isnull().any())
print("--------------------------------------------")
print("Number NA in %s: %d" % ("dfTrainY", dfTrainY.isnull().sum().sum()))
print("If NA exists in %s" % "dfTrainY")
print(dfTrainY.isnull().any())
print("--------------------------------------------")
print("Number NA in %s: %d" % ("dfTestX", dfTestX.isnull().sum().sum()))
print("If NA exists in %s" % "dfTestX")
print(dfTestX.isnull().any())
print("--------------------------------------------")
files = (dfTrainX, dfTrainY, dfTestX, dfTestId)
filenames = ("dfTrainX", "dfTrainY", "dfTestX", "dfTestId")
Data_proc.persist(config, files, filenames)
print("3.Feature Extraction Done")
print("========================================================")


##############################
## 4.Single Model Execution ##
##############################
#4.1 define feature set, model type, input/output path, log path, run times and cv fold and bagging times
#Note: run decide how to group data, fold decide which part to train and which part to test
#4.2 read input data by feature set
#4.3 read config info by model type
#4.4 write config and model type, feature set info to log based on log path
#4.5 read object function by model type
#4.6 define fmin
#4.7 get best predict and log out
#4.8 hence for each given feat set and model => get the best param

feat_name = "all"
model_list = ["param_space_reg_xgb_linear", "param_space_reg_xgb_tree", "param_space_cls_skl_rf"]
for model_name in model_list:
	best = model_hyperopt_tune(config, feat_name, model_name, loss_fct=None)
	print("################%s################/n%s" % (model_name, best))


##########################
## 5.Ensemble Selection ##
##########################
model_weight = ensemble_hyperopt_tune(config, ensemble_method="blending", model_list=model_list)


############################
## 6.Test Data Prediction ##
############################