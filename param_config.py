class ParamConfig:
	def __init__(self, train_path, test_path, output_folder, class_label, n_run, n_fold, n_bag, bagging_size, bagging_fraction):
		self.train_path = train_path
		self.test_path = test_path
		self.output_folder = output_folder
		self.class_label = class_label
		self.n_run = n_run
		self.n_fold = n_fold
		self.n_bag = n_bag
		self.bagging_size = bagging_size
		self.bagging_fraction = bagging_fraction
		
## initialize a param config
config = ParamConfig(train_path = "./titanic/train.csv",
					 test_path = "./titanic/test.csv",
					 output_folder = "./out/titanic",
					 class_label = 1,
					 n_run = 10,
					 n_fold = 10,
					 n_bag = 2, 
					 bagging_size = 3,
					 bagging_fraction = 0.9)

new_config = ParamConfig(train_path = "",
					 test_path = "",
					 output_folder = "./out/lung",
					 class_label = -1,
					 n_run = 5,
					 n_fold = 5,
					 n_bag = 2, 
					 bagging_size = 3,
					 bagging_fraction = 0.9)
