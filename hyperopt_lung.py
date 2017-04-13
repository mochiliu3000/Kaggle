import numpy as np
import pandas as pd
from param_config import new_config
from core.models.hyperopt_tune import model_hyperopt_tune
from sklearn.cross_validation import StratifiedKFold
from core.data_proc import Data_proc
from sklearn.decomposition import PCA


train_file = "/opt/LungCancerRecog/data/stage1_labels.csv"
test_file = "/opt/LungCancerRecog/data/stage1_sample_submission.csv"
feat_folder = "/opt/LungCancerRecog/out_origin_feat"
pca_run = True
n_comp = 60

def prepCV(config, dfTrain):	
	skf = [0]*config.n_run
	for run in range(config.n_run):
		seed = 2017 + 1000 * (run + 1)
		# each run, use different seed to split train/valid data
		skf[run] = StratifiedKFold(dfTrain, n_folds=config.n_fold, shuffle=True, random_state=seed)
		for fold, (trainInd, validInd) in enumerate(skf[run]):
			print("=================================")
			print("TrainInd for run: %d fold: %d" % (run+1, fold+1))
			print(trainInd[:5])
			print("ValidInd for run: %d fold: %d" % (run+1, fold+1))
			print(validInd[:5])
	print("INFO: CV prepared and stored successfully")
	return skf


def run_pca(feat):
    pca = PCA(n_components = n_comp)
    new_feat = pca.fit_transform(feat)
    print("INFO: ALL new feature has shape: " + str(new_feat.shape))
    print("INFO: PCA explains variance ratio: " + str(pca.explained_variance_ratio_))
    print("INFO: PCA totally explains variance: " + str(sum(pca.explained_variance_ratio_)))
    return new_feat


if __name__ == '__main__':
	#1. Generate training, test dataset and save them 
	df_train = pd.read_csv(train_file)
	df_test =  pd.read_csv(test_file)
	if pca_run:
		dfTrainX = run_pca(np.array([np.mean(np.load('%s/%s.npy' % (feat_folder, str(id))), axis=0) for id in df_train['id'].tolist()]))
		dfTestX = run_pca(np.array([np.mean(np.load('%s/%s.npy' % (feat_folder, str(id))), axis=0) for id in df_test['id'].tolist()]))
	else:
		dfTrainX = np.array([np.mean(np.load('%s/%s.npy' % (feat_folder, str(id))), axis=0) for id in df_train['id'].tolist()])
        	dfTestX = np.array([np.mean(np.load('%s/%s.npy' % (feat_folder, str(id))), axis=0) for id in df_test['id'].tolist()])
	dfTestId = df_test['id']
	print("INFO: dfTrainX has shape: " + str(dfTrainX.shape))
	print("INFO: dfTestX has shape: " + str(dfTestX.shape))
	dfTrainY = df_train['cancer'].as_matrix()
	files = (dfTrainX, dfTrainY, dfTestX, dfTestId)
	filenames = ("dfTrainX", "dfTrainY", "dfTestX", "dfTestId")
	Data_proc.persist(new_config, files, filenames)
	
	#2. Generate stratifiedKFold_ind
	skf = prepCV(new_config, dfTrainY)
	Data_proc.persist(new_config, (skf,), ("stratifiedKFold_ind",))
	
	
	#3. Set feature and models, run them
	feat_name = "all"
	model_list = ["param_space_reg_xgb_linear", "param_space_reg_xgb_tree", "param_space_cls_skl_rf"]
	#model_list = ["param_space_cls_skl_rf"]
	for model_name in model_list:
		best = model_hyperopt_tune(new_config, feat_name, model_name, loss_fct=None)
		print("################%s################/n%s" % (model_name, best))
