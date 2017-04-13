from hyperopt import fmin, tpe, STATUS_OK, Trials
import xgboost as xgb
import os
import pdb
from hyperopt import hp
from sklearn.datasets import load_svmlight_file
import numpy as np
import cPickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import *
#http://stackoverflow.com/questions/279237/import-a-module-from-a-relative-path
import sys, os 
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_proc import Data_proc

trial_counter = 0

def model_hyperopt_tune(config, feat_name, model_name, loss_fct):
	# TODO: put this data reading part here???, otherwise run again and again?????
	#feat_train_path = "C:/Mol/Data/output"
	#feat_valid_path = "C:/Mol/Data/output"
	#X_train, labels_train, X_valid, labels_valid = load_data("all", feat_train_path)	
	#n_train = X_train.shape[0]
	#n_valid = X_valid.shape[0]
	base_path = "%s/single_model" % config.output_folder
	params = params_gen(feat_name, model_name)
	trials = Trials()
	obj = lambda p: model_hyperopt_obj(config, feat_name, model_name, p, loss_fct)	
	best = fmin(obj, params, tpe.suggest, 200, trials)
	trial_kappas = np.asarray(trials.losses(), dtype=float)
	prec_cv_best = min(trial_kappas)
	print("Final Best: %.6f" % prec_cv_best)
	ind = np.where(trial_kappas == prec_cv_best)[0][0] + 1
	# log result, persist best and return best
	with open(base_path + "/" + model_name + "/best_ind", "wb") as f:
		cPickle.dump(ind, f, -1)
	return best

	
def model_hyperopt_obj(config, feat_name, model_name, params, loss_fct):
	#NOTE: This object function defines how to calculate the best loss for hyperopt, that is to select the best param set to maximize pred_cv_mean of valid datasets for a model_name*feat_name. During the fmin in model_hyperopt_tune, this object function will be run again and again to converge to the best param set. In a iteration, model_valid will be called once to calulate the average performance of this model_name*feat_name, this is the loss to maximize; In a iteration, model_test will be called once as well to do prediction on test dataset and write a csv file. Hence during the convergency, a series of pred_cv_mean and pred.csv will be generated and written out. During over-writing, only pred_cv_mean and pred.csv with the best param set will be left there in the out_folder for a model_name*feat_name. This will be utilized further for ensemble selections.  
	
	#1. set runs, folds and bags
	global trial_counter
	trial_counter += 1
	print("=========================== Trial %d ===========================" % trial_counter)
	n_run = config.n_run
	n_fold = config.n_fold
	n_bag = config.n_bag
	base_path = "%s/single_model" % config.output_folder
	#2. call model_valid
	pred_cv_mean = model_valid(config, feat_name, model_name, params, loss_fct, n_run, n_fold, n_bag)
	print("During the convergency, current pred_cv_mean is: %.6f" % pred_cv_mean)
	#3. call model_test
	model_test(config, feat_name, model_name, params, loss_fct, n_bag)
	#4. return loss for minimization
	#if pred_cv_mean > best:
		#best = pred_cv_mean
		#print("New best: %.6f" % best)
	return {'loss': pred_cv_mean, 'status': STATUS_OK}
	
	
def model_valid(config, feat_name, model_name, params, loss_fct, n_run, n_fold, n_bag):
	#NOTE: validation is done for valid dataset, and the best model is trained using Run*Fold*Bag datasets. This training datasets are smaller than the bagging datasets used in retraining section.
	
	base_path = "%s/single_model" % config.output_folder
	out_folder = "%s/%s" % (base_path, model_name)
	log_folder = ""
	prec_cv = np.zeros((n_run, n_fold), dtype=float)
	# TODO: put this data reading part outside model_run, otherwish run again and again?????
	
	skf = Data_proc.load(config, ("stratifiedKFold_ind",), ("skf",))["skf"]
	sub_dirs = ("dfTrainX", "dfTrainY")
	file = Data_proc.load(config, sub_dirs, ("dfTrainX", "dfTrainY"))
	dfTrainX = file["dfTrainX"] 
	dfTrainY = file["dfTrainY"] 
	#########################################################################
	for run in range(1, n_run + 1):
		for fold in range(1, n_fold + 1):
			for fold_num, (trainInd, validInd) in enumerate(skf[run-1]):
				if (fold-1) == fold_num:
					#trainInd = trainInd.tolist()
					#validInd = validInd.tolist()
					trainInd = map(int, trainInd)
					validInd = map(int, validInd)
					#X_train = dfTrainX.ix[trainInd,:]
					#X_valid = dfTrainX.ix[validInd,:]
                                        X_train = dfTrainX[trainInd,:]
                                        X_valid = dfTrainX[validInd,:]
					n_train = X_train.shape[0]
					n_valid = X_valid.shape[0]
					#http://stackoverflow.com/questions/33144039/typeerror-only-integer-arrays-with-one-element-can-be-converted-to-an-index-3
					#NOTE: all the input dataset needs to be numpy array. pandas dataframe has row index. When slice using X_train.ix[index_base,:], it will generate NA rows
					labels_train = np.array([dfTrainY[i] for i in trainInd])
					labels_valid = np.array([dfTrainY[i] for i in validInd])
					#X_train = X_train.as_matrix()
					#X_valid = X_valid.as_matrix()
		
			#1.set path and random state for each run & fold
			rdm = np.random.RandomState(2016 + 1000 * run + 10 * fold)
			out_path = "%s/Run%d/Fold%d" % (out_folder, run, fold)
			if model_name == "param_space_cls_skl_rf":
				booster = False
			else:
				booster = True
			#Note: windows 183 error, path exists will raise that error
			#if not os.path.exists(out_path):
				#os.makedirs(out_folder)
		
			#2.initial zero array to store predict values
			preds_bagging = np.zeros((n_valid, n_bag), dtype = float)
			
			#3.for each bagging iteration, use the same rdm but regenerate data for model fitting
			for n in range(1, n_bag + 1):
				#3.1.generate index for bagging with/without replacement
				index_base, index_meta = gen_index(rdm, n_train, replace=False, samp_ratio=0.95)		
				#3.2.xgboost need Matrix define with weight
				if booster:
					print("Running %s!" % model_name)
					dtrain_base, dvalid_base, watchlist = init_DMatrix(X_train, labels_train, X_valid, labels_valid, index_base)
				#3.3.load model
				model = load_model(model_name, params)
				
				if booster:
					#3.4.train model
					best_model = model.train(params, dtrain_base, params['num_round'], watchlist, feval=evalerror_regrank_valid)
					#3.5.predict model
					pred_valid = best_model.predict(dvalid_base)
				else:
					#3.4.train model
					rf = RandomForestClassifier(n_estimators=int(params['n_estimators']), criterion=params['criterion'], max_features=params['max_features'], max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'], n_jobs=params['n_jobs'], random_state=rdm)
					#rf = RandomForestClassifier(n_estimators=100)
					rf.fit(X_train[index_base,:], labels_train[index_base])
					print("++++++++++%s++++++++++" % model_name)
					print("Training Score: %.6f" % rf.score(X_train[index_base,:], labels_train[index_base]))
					#3.5.predict model
					pred_valid = rf.predict_proba(X_valid)[1]
					print("Testing Score: %.6f" % rf.score(X_valid, labels_valid))
					
				#3.6.store preds
				preds_bagging[:,n - 1] = pred_valid
				#3.7.use bagging mean as temp pred for current n iterations
				pred_temp_mean = np.mean(preds_bagging[:,:n], axis=1)
				#print(pred_temp_mean)	
				#3.8.a label method is needed here to label continue values to class label. May use cdf/pdf
				pred_temp_label = pred_temp_mean
                                print(pred_temp_label)
                                print("-----------------------------------------")
                                print(labels_valid)
				#if booster:
				#	pred_temp_label = label_score(pred_temp_mean)
				#print(pred_temp_label)	
				#3.9.need another method here to calculate the precision of your predition. May use quadratic weighted kappa
				#print(labels_valid)
				prec = get_prec(pred_temp_label, labels_valid)
				print(">>>>>> Precision for run %s, fold %s, bag %s: %.6f" % (run, fold, n, prec))
					
			#4.store final prec value for all bagging iterations into prec_cv within for each run & fold
			prec_cv[run - 1,fold - 1] = prec
			#print(prec_cv)			
			
			#5.persist temp mean, label and cv matrix for each run & fold
			#TODO: need to mkdir for run and fold folders
			if not os.path.exists(out_folder + "/run" + str(run) + "/fold" + str(fold)):
				os.makedirs(out_folder + "/run" + str(run) + "/fold" + str(fold))
			with open(out_folder + "/run" + str(run) + "/fold" + str(fold) + "/X_train", "wb") as f:
				cPickle.dump(X_train, f, -1)
			with open(out_folder + "/run" + str(run) + "/fold" + str(fold) + "/X_valid", "wb") as f:
				cPickle.dump(X_valid, f, -1)
			with open(out_folder + "/run" + str(run) + "/fold" + str(fold) + "/labels_train", "wb") as f:
				cPickle.dump(labels_train, f, -1)
			with open(out_folder + "/run" + str(run) + "/fold" + str(fold) + "/labels_valid", "wb") as f:
				cPickle.dump(labels_valid, f, -1)
			with open(out_folder + "/run" + str(run) + "/fold" + str(fold) + "/pred_model_mean", "wb") as f:
				cPickle.dump(pred_temp_mean, f, -1)
			with open(out_folder + "/run" + str(run) + "/fold" + str(fold) + "/pred_model_label", "wb") as f:
				cPickle.dump(pred_temp_label, f, -1)
			with open(out_folder + "/run" + str(run) + "/fold" + str(fold) + "/prec_cv", "wb") as f:
				cPickle.dump(prec_cv, f, -1)

	#########################################################################
	pred_cv_mean = np.mean(prec_cv)
	pred_cv_std = np.std(prec_cv)
	print("pred_cv_mean: %.6f" % pred_cv_mean)
	print("==================================================================")
	mean_raw_path = out_folder + "/mean" 
	std_raw_path = out_folder + "/std"
	if not os.path.exists(mean_raw_path):
                os.makedirs(mean_raw_path)
	if not os.path.exists(std_raw_path):
                os.makedirs(std_raw_path)
	with open(mean_raw_path + "/pred_cv_mean_" + str(trial_counter), "wb") as f:
		cPickle.dump(pred_cv_mean, f, -1)
	with open(std_raw_path + "/pred_cv_std_" + str(trial_counter), "wb") as f:
		cPickle.dump(pred_cv_std, f, -1)
	return pred_cv_mean


def model_test(config, feat_name, model_name, params, loss_fct, n_bag):
	#NOTE: prediction is done for test dataset, and the best model is retrained using bagging datasets. NOT using the best model and param trained by Run*Fold*Bag datasets in training section. This retraining bagging datasets are bigger than the Run*Fold*Bag datasets. 
	
	#1. init RandomState, input dataset, out_folder...
	rdm = np.random.RandomState(2016 + 1000 * n_bag)
	base_path = "%s/single_model" % config.output_folder 
	log_folder = ""	
	sub_dirs = ("dfTrainX", "dfTestX", "dfTrainY", "dfTestId")
	file = Data_proc.load(config, sub_dirs, ("dfTrainX", "dfTestX", "dfTrainY", "dfTestId"))
	X_train = file["dfTrainX"]
	labels_train = file["dfTrainY"]
	X_test = file["dfTestX"]
	id_test = file["dfTestId"]

	#X_train = X_train.as_matrix()
	#X_test = X_test.as_matrix()
	
	if model_name == "param_space_cls_skl_rf":
		booster = False
	else:
		booster = True
		
	#2. init a zero matrix to store bagging preds
	n_train = X_train.shape[0]
	n_test = X_test.shape[0]
	preds_bagging = np.zeros((n_test, n_bag), dtype=float)
	#3. for each bag, retrain best model and do prediction
	for n in range(n_bag):	
		#3.1.generate index for bagging with/without replacement
		index_base, index_meta = gen_index(rdm, n_train, replace=False, samp_ratio=0.95)		
		#3.2.xgboost needs Matrix define with weight, currently set default weight for each data point as 1 (equally weighted)
		if booster:
			print("Running %s!" % model_name)
			dtrain_base, dtest_base, watchlist = init_DMatrix(X_train, labels_train, X_test, None, index_base)
		#3.3.load model
		model = load_model(model_name, params)
		if booster:
			#3.4.train model
			best_model = model.train(params, dtrain_base, params['num_round'], watchlist, feval=evalerror_regrank_test)
			#3.5.predict model
			pred_test = best_model.predict(dtest_base)
		else:
			#3.4.train model
			rf = RandomForestClassifier(n_estimators=int(params['n_estimators']), criterion=params['criterion'], max_features=params['max_features'], max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'], n_jobs=params['n_jobs'], random_state=rdm)
			#rf = RandomForestClassifier(n_estimators=100)
			rf.fit(X_train[index_base,:], labels_train[index_base])
			print("++++++++++%s++++++++++" % model_name)
			print("Training Score: %.6f" % rf.score(X_train[index_base,:], labels_train[index_base]))
			#3.5.predict model
			pred_test = rf.predict_proba(X_test)[1]
		#3.6.store pred_test
		preds_bagging[:,n] = pred_test
		
	#4. persist mean(preds_bagging)
	pred_raw_path = "%s/%s/%s" % (base_path, model_name, "test")
	if not os.path.exists(pred_raw_path):
		os.makedirs(pred_raw_path) 
	pred_raw = np.mean(preds_bagging, axis=1)
	pred_out = pd.DataFrame({"id": id_test, "cancer": pred_raw})
	pred_out.to_csv("%s/pred_%s.csv" % (pred_raw_path, trial_counter), index=False)
	print("During the convergency, write out current pred.csv to %s" % pred_raw_path)
	print("------------------------------------------------------------------")


###################### Utils funtions below ######################
def params_gen(feat_name, model_name):	
	#params = params_space[model_name]
	#TODO: make this better
	if model_name == "param_space_reg_xgb_linear":
		params = param_space_reg_xgb_linear
	if model_name == "param_space_reg_xgb_tree":
		params = param_space_reg_xgb_tree
	if model_name == "param_space_cls_skl_rf":
		params = param_space_cls_skl_rf
	return params
	
	
def gen_index(rdm, n_train, replace, samp_ratio):
	n_samp = int(n_train * samp_ratio)
	if replace:				
		index_base = rdm.randint(n_train, size=n_samp)
		index_meta = [i for i in range(n_train) if i not in index_base]
	else:
		rand_num = rdm.uniform(size=n_train)
		index_base = [i for i in range(n_train) if rand_num[i] <= samp_ratio]
		index_meta = [i for i in range(n_train) if rand_num[i] > samp_ratio]
	return index_base, index_meta


def init_DMatrix(X_train, labels_train, X_pred, labels_pred, index_base):
	#http://stackoverflow.com/questions/15891038/pandas-change-data-type-of-columns
	#X_valid = X_valid.apply(lambda x: pd.to_numeric(x, errors='ignore'))
	#X_train = X_train.apply(lambda x: pd.to_numeric(x, errors='ignore'))
	dtrain_base = xgb.DMatrix(X_train[index_base,:], label=labels_train[index_base])
	watchlist = []
	if labels_pred is None:
		dpred_base = xgb.DMatrix(X_pred)
	else:
		dpred_base = xgb.DMatrix(X_pred, label=labels_pred)
	return dtrain_base, dpred_base, watchlist

	
def load_data(feat_name, feat_path):
	#train = load_svmlight_file(feat_path + "\dfTrainX")
	#labels_train = load_svmlight_file(feat_path + "\dfTrainY")
	#valid = load_svmlight_file(feat_path + "\dfTestX")
	#labels_valid = load_svmlight_file(feat_path + "\dfTestY")
	dfTrainX_file = open(feat_path + "/dfTrainX", 'rb').read()  
	train = cPickle.loads(dfTrainX_file)  
	dfTrainY_file = open(feat_path + "/dfTrainY", 'rb').read()  
	labels_train = cPickle.loads(dfTrainY_file)
	dfTestX_file = open(feat_path + "/dfTestX", 'rb').read()  
	valid = cPickle.loads(dfTestX_file)
	dfTestY_file = open(feat_path + "/dfTestY", 'rb').read()  
	labels_valid = cPickle.loads(dfTestY_file)
	return train, np.array(labels_train), valid, np.array(labels_valid)
	
	
def load_model(model_name, params):
	#model = getattr(models, model_name)
	model = xgb
	#model = RandomForestClassifier	
	return model
	
	
def evalerror_regrank(preds, dtrain):
	## labels are in [0,1]
	labels = dtrain.get_label()
	#pred_temp_label = label_score(preds)
	#kappa = quadratic_weighted_kappa(labels, preds)
	#kappa = precision(labels, preds)
	kappa = get_prec(labels, preds)
	## we return -kappa for using early stopping
	#kappa *= -1.
	return 'kappa', float(kappa)
	
# TODO: is it right to use the cdf of all training dataset to calculate threshold for both valid and test preds ??? 
evalerror_regrank_valid = lambda preds, dtrain: evalerror_regrank(preds, dtrain)		
evalerror_regrank_test = lambda preds, dtrain: evalerror_regrank(preds, dtrain)
	
	
###################### Params Space below ######################	
#http://iopscience.iop.org/article/10.1088/1749-4699/8/1/014008
#hp.quniform(label, low, high, q). Drawn by round(uniform(low, high) / q) * q, Suitable for a discrete value with respect to which the objective is still somewhat smooth.
param_space_reg_xgb_linear = {
    'task': 'classify',
    'booster': 'gblinear',
    'objective': 'binary:logistic',
    'eta' : hp.quniform('eta', 0.01, 1, 0.01),
    'lambda' : hp.quniform('lambda', 0.1, 5, 0.05),
    'alpha' : hp.quniform('alpha', 0.15, 0.5, 0.005),
    'lambda_bias' : hp.quniform('lambda_bias', 0, 3, 0.1),
    'num_round' : 300,	
    #'num_round' : hp.quniform('num_round', 10, 500, 5),	
    'nthread': 10,
    'silent' : 1,
    'seed': 2017,
    "max_evals": 20,
}

param_space_reg_xgb_tree = {
    'task': 'classify',
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eta': hp.quniform('eta', 0.01, 1, 0.01),
    'gamma': hp.quniform('gamma', 0, 2, 0.1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'max_depth': 4,
    #'max_depth': hp.quniform('max_depth', 1, 10, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.1),
    #'num_round': hp.quniform('num_round', xgb_min_num_round, xgb_max_num_round, xgb_num_round_step),
    'num_round': 200,
    'nthread': 10,
    'silent': 1,
    'seed': 2017,
    "max_evals": 20,
}

param_space_cls_skl_rf = {
    'task': 'classify',
    'objective': 'rf',
    'n_estimators': hp.quniform('n_estimators', 100, 500, 50),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    'max_features': hp.quniform("max_features", 0.5, 1.0, 0.1),
    'n_jobs': 10,
    'max_depth': hp.quniform('max_depth', 1, 3, 1),
    'min_samples_split': hp.quniform('min_samples_split', 1, 3, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 3, 1),
    'max_evals': 50,
}
