from data_proc import Data_proc
from param_config import config
import numpy as np
import numbers
import re
import pandas as pd

class Feat_extr:
	#TODO: for labels = [-1,], make series to tuple, may think better way to work on series not tuple
	@staticmethod
	def transform(dfTrain, methods, dfTest=None, labels=[-1,]):	
		if len(methods) == 0:
			print("Warn: No transforming method sepecified")
		else:
			if len(methods) !=  len(labels):
				print("Error: Transforming dfTrainX, dfTestX error, labels and methods are mis-matching")
			else:
				for i in range(len(labels)):
					if labels[i] == -1:
						j = 0
					else:
						j = i	
					for k in range(len(methods[j])):	
						# http://stackoverflow.com/questions/7936572/python-call-a-function-from-string-namemethod
						method = getattr(Feat_extr, methods[j][k])
						if len(dfTrain.shape) == 1:
							dfTrain = method(dfTrain)
							if dfTest is not None:
								dfTest = method(dfTest)
						else:
							dfTrain.ix[:,labels[j]] = method(dfTrain.ix[:,labels[j]])
							if dfTest is not None:
								dfTest.ix[:,labels[j]] = method(dfTest.ix[:,labels[j]])
		return dfTrain, dfTest
	
	@staticmethod
	def combine(dfTrain, methods, labels, dfTest=None):
		if len(methods) == 0 or len(labels) == 0 or len(methods) != len(labels):
			print("Warn: Number of methods or labels is incorrect")
		else:
			for i in range(len(methods)):
				for j in range(len(methods[i])):
					method = getattr(Feat_extr, methods[i][j])
					#combine 2 cols and fill it into the 1st col, remove 2nd col
					dfTrain.ix[:labels[i][0]] = method(dfTrain.ix[:labels[i][0]], dfTrain.ix[:labels[i][1]])
					dfTrain = dfTrain.drop(dfTrain.columns[[labels[i][1]]], axis=1)
					if dfTest is not None:
						dfTest.ix[:labels[i][0]] = method(dfTest.ix[:labels[i][0]], dfTest.ix[:labels[i][1]])
						dfTest = dfTest.drop(dfTest.columns[[labels[i][1]]], axis=1)
		return dfTrain, dfTest
			
	@staticmethod		
	def drop(dfTrainX, dfTestX, labels):
		dfTrainX = dfTrainX.drop(dfTrainX.columns[[labels]], axis=1)
		dfTestX = dfTestX.drop(dfTestX.columns[[labels]], axis=1)
		return dfTrainX, dfTestX
			
	# TODO:Private methods
	#@staticmethod	
	#def pca(col):
	
	@staticmethod	
	def log(col):
		# http://stackoverflow.com/questions/4187185/how-can-i-check-if-my-python-object-is-a-number
		if all([isinstance(item, numbers.Number) and item > 0 for item in col]):
			# http://stackoverflow.com/questions/23748842/understanding-math-errors-in-pandas-dataframess
			return np.log(np.float64(col))
		else:
			print("Error: Not all items for log are positive numbers")
			return col
	
	@staticmethod
	def to_value(col):
		dict = {}
		j = 0
		for i in range(len(col)):
			if col[i] in dict:
				col[i] = dict[col[i]]
			else:
				dict[col[i]] = j
				col[i] = j
				j += 1
		Data_proc.persist(config, (dict,), ("dict_to_value",))
		return col
		
		
	@staticmethod
	def extract(col):
		pattern = re.compile('.*, *([A-Za-z ]+)\.')
		f = lambda i: pattern.match(i).group(1) 
		return col.apply(f)
		
	@staticmethod
	def map_with_dict(col):
		dict = {"Capt": "O", "Col": "O", "Major": "O", "Jonkheer": "R", "Don": "R", "Sir": "R", "Dr": "O", "Rev": "O", "the Countess": "R", "Dona": "R", "Mme": "Mrs", "Mlle": "Miss", "Ms": "Mrs", "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master", "Lady": "R"}
		f = lambda i: dict[i]
		return col.apply(f)
		
	@staticmethod
	def comb_add(col1, col2):
		print("Combining cols !!!!!!!!!!!!!!!!!!!!!!!")
		print(col1)
		print(col2)
		return col1 + col2