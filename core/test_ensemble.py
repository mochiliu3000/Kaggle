import pdb
from ensemble import *
import cPickle

#pdb.set_trace()

model_weight = ensemble_hyperopt_tune(ensemble_method="blending", model_list=["param_space_reg_xgb_linear", "param_space_reg_xgb_tree", "param_space_cls_skl_rf"], bagging_size=3, bagging_fraction=0.9)



'''
m1 = "C:/Mol/titanic/single_model/param_space_cls_skl_rf/run1/fold1/pred_model_mean"

m2 = "C:/Mol/titanic/single_model/param_space_reg_xgb_linear/run1/fold1/pred_model_mean"

m3 = "C:/Mol/titanic/single_model/param_space_reg_xgb_tree/run1/fold1/pred_model_mean"

m4 = "C:/Mol/titanic/single_model/param_space_reg_xgb_tree/run1/fold1/labels_valid"

with open(m1, 'rb') as f:
	true_label_all = cPickle.load(f)
	print(true_label_all)
	
with open(m2, 'rb') as f:
	true_label_all = cPickle.load(f)
	print(true_label_all)
	
with open(m3, 'rb') as f:
	true_label_all = cPickle.load(f)
	print(true_label_all)

with open(m4, 'rb') as f:
	true_label_all = cPickle.load(f)
	print(true_label_all)
'''