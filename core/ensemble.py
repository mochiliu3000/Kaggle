import operator
from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, Trials
import os
import numpy as np
import pandas as pd
import cPickle
import pickle
from models.utils import *


def ensemble_hyperopt_tune(config, ensemble_method, model_list):
	bagging_size = config.bagging_size
	bagging_fraction = config.bagging_fraction
	base_path = "%s/ensemble/blending" % config.output_folder
	test_path = "%s/test" % base_path
	best_bagged_model_list = ensemble_valid(config, ensemble_method, bagging_size, bagging_fraction, model_list)
	output = ensemble_test(config, best_bagged_model_list)
	if not os.path.exists(test_path):
		os.makedirs(test_path)
	submit_path = "%s/%s" % (test_path, "pred_ensemble.csv")
	output.to_csv(submit_path, index=False)
	print("Ensemble -- %s complete, persist final pred result into: %s" % (ensemble_method, submit_path))

def ensemble_valid(config, ensemble_method, bagging_size, bagging_fraction, model_list, max_iter=10):
	#1. init an empty list to store best model_weight. Each of them is a dict. The size of this list equals bagging_size.
	base_path = "%s/ensemble/blending" % config.output_folder
	best_bagged_model_list = []
	params = params_gen(ensemble_method)
	#2. for each bagging_iter, randomly select part of the models from model_list, then run greedy blending ensemble selection on these models involved. Store best model_weight of these models involved into best_bagged_model_list.
	for bagging_iter in range(bagging_size):
		#NOTE: suppose you have top 10 models, and within each bagging iteration not all models will be involved. Say for bagging_iter=0, M1-M3 are involved, for bagging_iter=1, M2-M4 are involved. Then for bagging_iter=0, W1-W3 are calculated based on greedy blending on valid dataset; while for M2-M4, W2'-W4' are calculated. An AVG of these 2 bags of weighted preds can be the final preds for test dataset.
		n_model = len(model_list)
		rng = np.random.RandomState(2016 + 100 * bagging_iter)
		randnum = rng.uniform(size=n_model)
		index_base = [i for i in range(n_model) if randnum[i] < bagging_fraction]
		model_involved = [model_list[i] for i in index_base]
		print("++++++++++++++++++++++++++++++++++bag%s++++++++++++++++++++++++++++++++++" % (bagging_iter+1))
		print("Model Involved: " + str(model_involved))
		model_involved_sorted, prec_cv_init, pred_list_temp, model_pred_dict, model_true_label, model_num_true_label = blending_init(config, model_involved, n_run=10, n_fold=10)
		print("========================================================================")
		
		flag = True
		model_weight = dict()
		for model in model_involved:
			model_weight[model] = 1.0
		iter = 0
		#NOTE: To avoid local max, re-run blending for max_iter times
		while True:
			iter += 1
			if iter > max_iter:
				print("Iterate for %d times, and reach the max_iter" % (iter-1))
				if flag:
					print("WARN: No weight has been modified")
				break
			for model in model_involved:
				print("------------------iter %d & model %s------------------" % (iter, model))
				print(model_weight)
				trials = Trials()				
				obj = lambda p: ensemble_hyperopt_obj(p, pred_list_temp, 1.0, model_pred_dict[model], model_true_label[model], model_num_true_label[model])
				best = fmin(obj, params, tpe.suggest, 100, trials)
				best_w2 = best['w2']
				print(best_w2)
				fnvals = [-t['result']['loss'] for t in trials.trials]
				prec_cv_new = max(fnvals)
				ind_max = fnvals.index(max(fnvals))
				pred_attach = trials.trial_attachments(trials.trials[ind_max])['this_pred']
				best_pred = pickle.loads(pred_attach)
				if prec_cv_new >= prec_cv_init:
					prec_cv_init = prec_cv_new
					model_weight[model] += best_w2
					#update the pred_list_temp with new best weight
					pred_list_temp = best_pred
					print("During iteration %d, modify weight of model: %s" % (iter, model))
					print("New prec_cv: %.6f" % prec_cv_init)
					flag = False
		print("========================================================================")
		print("Final model_weights: %s" % str(model_weight))
		print("Final model precision: %.6f" % prec_cv_init)
		
		# log result, persist best and save best
		best_bagged_model_list.append(model_weight)
		out_folder = "%s/bag%s" % (base_path, (bagging_iter+1))
		if not os.path.exists(out_folder):
			os.makedirs(out_folder)
		with open(out_folder + "/prec_cv_init", "wb") as f:
			cPickle.dump(prec_cv_init, f, -1)
		with open(out_folder + "/model_weight", "wb") as f:
			cPickle.dump(model_weight, f, -1)
	#3. return best_bagged_model_list.	
	return best_bagged_model_list

def ensemble_test(config, best_bagged_model_list):
	model_folder = "%s/single_model" % config.output_folder
	bagging_size = len(best_bagged_model_list)
	
	#TODO: some of the data here already labeled, e.g. rf. Don't use weighted avg on them here, may use primary vote!!! Or label those needed, then primary vote with already labeled ones
	for bagging_iter in range(bagging_size):
		w_ens = 0
		iter = 0
		for model_name,w in best_bagged_model_list[bagging_iter].iteritems():
			iter += 1
			pred_file = "%s/%s/test/pred.csv" % (model_folder, model_name)
			this_p_test = pd.read_csv(pred_file, dtype=float)["prediction"].values
			this_w = w
			if iter == 1:
				p_ens_test = np.zeros((this_p_test.shape[0]),dtype=float)
				id_test = pd.read_csv(pred_file, dtype=float)["id"].values
				id_test = np.asarray(id_test, dtype=int)
			p_ens_test = (w_ens * p_ens_test + this_w * this_p_test) / (w_ens + this_w)
			w_ens += this_w
		if bagging_iter == 0:
			p_ens_test_bag = p_ens_test
		else:
			p_ens_test_bag = (bagging_iter * p_ens_test_bag + p_ens_test) / (bagging_iter+1.)
		
	p_ens_test_bag = label_score(p_ens_test_bag)
	output = pd.DataFrame({"id": id_test, "prediction": p_ens_test_bag})
	return output

def ensemble_hyperopt_obj(params, p1_list, w1, p2_list, true_label, num_true_label):
	n_run=10
	n_fold=10
	w2 = params['w2']
	base_folder = "C:/Mol/titanic"
	prec_cv = np.zeros((n_run, n_fold), dtype=float)
	with open("%s/dfTrainY" % base_folder, 'rb') as f:
		true_label_all = cPickle.load(f)
	n_valid_all = len(true_label_all)
	this_pred = np.zeros((n_run, n_fold, n_valid_all), dtype=float)
	for run in range(n_run):
		for fold in range(n_fold):
			#http://stackoverflow.com/questions/34095946/python-3-5-numpy-how-to-avoid-deprecated-technique
			this_num_true_label = int(num_true_label[run][fold])
			p1 = p1_list[run][fold][:this_num_true_label]
			p2 = p2_list[run][fold][:this_num_true_label]
			this_true_label = true_label[run][fold][:this_num_true_label]
			p_ens = (w1 * p1 + w2 * p2) / (w1 + w2)
			this_pred[run, fold, :this_num_true_label] = p_ens
			p_ens_label = label_score(p_ens)
			prec = get_prec(p_ens_label, this_true_label)
			prec_cv[run][fold] = prec
	pred_cv_mean = np.mean(prec_cv)
	return {'loss': -pred_cv_mean, 'status': STATUS_OK, 'attachments': {'this_pred': pickle.dumps(this_pred)}}

	
def blending_init(config, model_list, n_run, n_fold):
	base_path = "%s/ensemble/blending" % config.output_folder
	if not os.path.exists(base_path):
		os.makedirs(base_path)
	model_folder = "%s/single_model" % config.output_folder
	log_folder = ""
	model_pred_dict = dict()
	model_prec_dict = dict()
	model_true_label = dict()
	model_num_true_label = dict()
	n_model = len(model_list)
	
	# 1. load models
	for model in model_list:
		prec_cv = np.zeros((n_run, n_fold), dtype=float)
		#NOTE: each run/fold has different numbers of training/valid data, here we initial it with all the training data
		with open("%s/dfTrainY" % config.output_folder, 'rb') as f:
			true_label_all = cPickle.load(f)		
		n_valid_all = len(true_label_all)
		pred_list_valid = np.zeros((n_run, n_fold, n_valid_all), dtype=float)
		true_label = np.zeros((n_run, n_fold, n_valid_all), dtype=float)
		num_true_label = np.zeros((n_run, n_fold, 1), dtype=int)
		for run in range(1, n_run + 1):
			for fold in range(1, n_fold + 1):
				
				#read preds for each run & fold
				with open(model_folder + "/" + model + "/run" + str(run) + "/fold" + str(fold) + "/pred_model_mean", 'rb') as f:
					this_pred_model_mean = cPickle.load(f)
				#read precs for each run & fol
				with open(model_folder + "/" + model + "/run" + str(run) + "/fold" + str(fold) + "/prec_cv", 'rb') as f:
					this_prec_model = cPickle.load(f)[run-1][fold-1]
				#read true_label for each run & fold
				with open(model_folder + "/" + model + "/run" + str(run) + "/fold" + str(fold) + "/labels_valid", 'rb') as f:
					this_true_label = cPickle.load(f)
				
				#store pred into pred_list_valid
				pred_list_valid[run-1][fold-1][:len(this_true_label)] = this_pred_model_mean
				#store prec into prec_cv
				prec_cv[run-1][fold-1] = this_prec_model
				#store true_label into true_label
				true_label[run-1][fold-1][:len(this_true_label)] = this_true_label
				num_true_label[run-1][fold-1] = len(this_true_label)
				
		model_pred_dict[model] = pred_list_valid
		model_prec_dict[model] = np.mean(prec_cv)
		model_true_label[model] = true_label
		model_num_true_label[model]= num_true_label
	
	# 2. sort model_pred_dict by prec
	sorted_models = sorted(model_prec_dict.items(), key=operator.itemgetter(1), reverse=True)
	
	# 3. ensemble - blending - init (greedy)
	w_ens, this_w = 0, 1.0
	pred_list_temp = np.zeros((n_run, n_fold, n_valid_all), dtype=float)
	ind = 0
	for model, prec_mean in sorted_models:
		prec_cv = np.zeros((n_run, n_fold), dtype=float)
		print(">>>>>> Add into ensemble the following model: %s" % model)
		print("It's prec_mean: %.6f" % prec_mean)
		for run in range(n_run):
			for fold in range(n_fold):
				#http://stackoverflow.com/questions/34095946/python-3-5-numpy-how-to-avoid-deprecated-technique
				this_num_true_label = int(model_num_true_label[model][run, fold])
				this_valid_pred = model_pred_dict[model][run, fold, :this_num_true_label]
				pred_list_temp[run, fold, :this_num_true_label] = (w_ens * pred_list_temp[run, fold, :this_num_true_label] + this_w * this_valid_pred) / (w_ens + this_w)
				if ind == n_model-1:				
					this_true_label = model_true_label[model][run, fold, :this_num_true_label]
					if model == "param_space_cls_skl_rf":
						this_pred_label = pred_list_temp[run, fold, :this_num_true_label]
					else:
						this_pred_label = label_score(pred_list_temp[run, fold, :this_num_true_label])
					prec = get_prec(this_pred_label, this_true_label)
					prec_cv[run][fold] = prec
		w_ens += 1.0
		ind += 1
	print("Init ends in %d iterations" % ind)
	print("Init weights: %s" % str((1.0,)*n_model))
	print("Init prec mean: %.6f std: %.6f" % (np.mean(prec_cv), np.std(prec_cv)))
	return sorted_models, np.mean(prec_cv), pred_list_temp, model_pred_dict, model_true_label, model_num_true_label
	

###################### Utils funtions below ######################	
def params_gen(ensemble_method):	
	#params = params_space[model_name]
	params = blending_param
	#params = param_space_reg_xgb_tree
	return params
	
	
###################### Params Space below ######################	
blending_param = {
    'w2': hp.uniform('w2', -1.0, 1.0)
}