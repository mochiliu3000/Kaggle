import pdb
from models.hyperopt_tune import *

#pdb.set_trace()
feat_name = "all"
model_name = "param_space_reg_xgb_linear"
#model_name = "param_space_reg_xgb_tree"
#model_name = "param_space_cls_skl_rf"

best = model_hyperopt_tune(feat_name, model_name, loss_fct=None)
print(best)


# import pdb
# import os; os.getcwd()
# os.chdir('C:\Users\IBM_ADMIN\Desktop\Kaggle\src\core')
# os.chdir('C:\Users\IBM_ADMIN\Desktop\Kaggle\src\core\models')
# import hyperopt_tune
# pdb.run()

#{'lambda_bias': 2.1, 'alpha': 0.275, 'eta': 0.23, 'lambda': 1.4500000000000002}