import pandas as pd
import numpy as np
from scipy import stats
import cPickle
import os
from sklearn.cross_validation import StratifiedKFold

class Data_proc:	
	@staticmethod
	def input(config, header=None):		
		dfTrain = pd.read_csv(config.train_path, header=header)
		dfTestX = pd.read_csv(config.test_path, header=header)
		dfTrainX = dfTrain.drop(dfTrain.columns[[config.class_label]], axis=1)	
		dfTrainY = dfTrain.ix[:,config.class_label]	
		return dfTrainX, dfTrainY, dfTestX
	
## TODO: basicly, scale, categorize, na_manage are all apply method and can be one	
	@staticmethod
	def scale(dfTrainX, dfTestX, labels, methods):
		if len(labels) != len(methods):
			print("Error: The length of labels is not equal to the length of methods")
			return 
		else:
			for i in range(len(labels)):
				if methods[i] == "__normalize":
					dfTrainX.ix[:,labels[i]] = Data_proc.__normalize(dfTrainX.ix[:,labels[i]])
					dfTestX.ix[:,labels[i]] = Data_proc.__normalize(dfTestX.ix[:,labels[i]])
				else:
					print("Warn: No scaling method has been applied for col "+ str(i))
		return dfTrainX, dfTestX
		
		
	@staticmethod
	def categorize(dfTrain, dfTest, labels, methods):
		if len(labels) != len(methods):
			print("Error: The length of labels is not equal to the length of methods")
			return
		else:
			for i in range(len(labels)):
				method = getattr(Data_proc, methods[i])
				dfTrain.ix[:,labels[i]] = method(dfTrain.ix[:,labels[i]])
				dfTest.ix[:,labels[i]] = method(dfTest.ix[:,labels[i]])
			return dfTrain, dfTest
	
	
	@staticmethod	
	def na_manage(dfTrain, dfTest, labels, methods):
		if len(labels) != len(methods):
			print("Error: The length of labels is not equal to the length of methods")
			return
		else:
			for i in range(len(labels)):
				method = getattr(Data_proc, methods[i])
				dfTrain.ix[:,labels[i]] = method(dfTrain.ix[:,labels[i]])
				dfTest.ix[:,labels[i]] = method(dfTest.ix[:,labels[i]])
			return dfTrain, dfTest
		
		
		
	@staticmethod	
	def persist(config, objects, filenames):
		if not os.path.exists(config.output_folder):
			os.makedirs(config.output_folder)
		for i in range(len(objects)):
			with open(config.output_folder + "/" + filenames[i], "wb") as f:
				cPickle.dump(objects[i], f, -1)
	
	@staticmethod
	def load(config, sub_dirs, filenames):
	#TODO:make sure len equal, and no duplicate items in filenames
		files = dict()
		for i in range(len(sub_dirs)):
			file_dir = "%s/%s" % (config.output_folder, sub_dirs[i])
			with open(file_dir, "rb") as f:
				file = cPickle.load(f)
			files[filenames[i]] = file
		return files
	
	@staticmethod
	def prepCV(config, header=None):
		dfTrain_origin = pd.read_csv(config.train_path, header=header)
		
		skf = [0]*config.n_run
		for run in range(config.n_run):
			seed = 2016 + 1000 * (run + 1)
			# each run, use different seed to split train/valid data
			skf[run] = StratifiedKFold(dfTrain_origin.ix[:,config.class_label-1], n_folds=config.n_fold, shuffle=True, random_state=seed)
			for fold, (trainInd, validInd) in enumerate(skf[run]):
				print("=================================")
				print("TrainInd for run: %d fold: %d" % (run+1, fold+1))
				print(trainInd[:5])
				print("ValidInd for run: %d fold: %d" % (run+1, fold+1))
				print(validInd[:5])
		print("INFO: CV prepared and stored successfully")
		return skf
		
		
	
	# TODO:Private methods
	@staticmethod
	def __normalize(col):
		col_mean = np.mean(col)
		col_std = np.std(col)
		return (col-col_mean)/col_std
			
	@staticmethod	
	def fill_median(col):
		col_median = np.median(col.dropna())
		if len(col[col.isnull()]) > 0:
			col[col.isnull()] = col_median
		else:
			print("WARN: NA not found when fill_median")
		return col
	
	@staticmethod	
	def fill_mode(col):
		col_mode = stats.mode(col.dropna())
		if len(col[col.isnull()]) > 0:
			col[col.isnull()] = col_mode
		else:
			print("WARN: NA not found when fill_mode")
		return col
		
	@staticmethod
	def has_char(col):
		for i in range(len(col)):
			try:
				num = int(col[i])
				col[i] = 0
			except Exception as err: 
				col[i] = 1
		return col