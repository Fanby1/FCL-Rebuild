import dataloaders
import os, copy
from torch.utils.data import random_split
from trainer import Trainer
from torch.utils.data import Subset

from utils.utils import write_dict_to_file
from utils.utils import federated_average

class FedTrainer:
	def __init__(self, args, seed: int, metric_keys, save_keys, client_num, attacker_num, 
			task_count: int, private_class_num, client_count: int, comunication_round_count: int, 
   			synchronize: bool = True, path: str = None):
		self.args = args
		self.task_count = task_count
		self.client_count = client_count
		self.comunication_round_count = comunication_round_count
		self.seed = seed
  
		self.metric_keys = metric_keys
		self.save_keys = save_keys
  
  
		if args.dataset == 'CIFAR100':
			DataSpliter = dataloaders.Cifar100_Spliter
			input_size = 224
		elif args.dataset == 'ImageNet_R':
			DataSpliter = dataloaders.ImageNetR_Spliter
			input_size = 224
		else:
			raise ValueError('Dataset not implemented!')
			
		# create spliter
		spliter = DataSpliter(client_num=client_num, attacker_num=attacker_num, task_num=task_count, 
						private_class_num=private_class_num, input_size=input_size, path=path)

		# split dataset
		if synchronize:
			self.client_subset, self.client_mask = spliter.random_split_synchron()
		else:
			self.client_subset, self.client_mask = spliter.random_split()
		self.surro_data, self.test_data = spliter.process_testdata(5)
		print(self.client_mask)

		self.split_data()
		self.split_global_data()
		  
		self.init_metric()
		self.init_trainner()
  
	def split_data(self):
		self.client_data_train = []
		self.client_data_val = []
		for subset in self.client_subset:
			temp_train = []
			temp_val = []
			for data in subset:
				# data, _ = random_split(data, [int(len(data) * 0.1), len(data) - int(len(data) * 0.1)])
				train_dataset, val_dataset = random_split(data, [int(len(data) * 0.7), len(data) - int(len(data) * 0.7)])
				temp_train.append(train_dataset)
				temp_val.append(val_dataset)
			self.client_data_train.append(temp_train)
			self.client_data_val.append(temp_val)
   
	def split_global_data(self):
		# culculate global data and mask
		self.global_datas = []
		self.global_masks = []
		for i in range(self.task_count):
			self.global_masks.append([])
			self.global_datas.append([])
			for j in range(self.client_count):
				self.global_masks[i].extend(self.client_mask[j][i])
			self.global_masks[i] = list(set(self.global_masks[i]))

			for idx in range(len(self.test_data.indices)):
				if self.test_data.targets[idx] in self.global_masks[i]:
					self.global_datas[i].append(idx)
			self.global_datas[i] = Subset(self.test_data, self.global_datas[i])
   
	def init_metric(self):
		for i in range(self.client_count):
			self.save_keys.append(f'client-{i + 1}')

		self.avg_metrics = {}
		for mkey in self.metric_keys: 
			self.avg_metrics[mkey] = {}
			for skey in self.save_keys: 
				self.avg_metrics[mkey][skey] = [None] * self.task_count
   
	def init_trainner(self):
		self.learner_pool = []
		self.client_weight = [1] * self.client_count
				
		self.global_trainer = Trainer(self.args, self.seed, self.metric_keys, self.save_keys, 
								train_dataset=self.surro_data, test_dataset=self.test_data,
								validate_dataset=self.global_datas, client_index=0, 
								class_mask=self.global_masks)


		for i in range(self.client_count):
			trainer = Trainer(self.args, self.seed, self.metric_keys, self.save_keys, 
					train_dataset=self.client_data_train[i], test_dataset=self.test_data, 
					validate_dataset=self.client_data_val[i], client_index=1 + i, 
	 				class_mask=self.client_mask[i])
			self.learner_pool.append(trainer)
	
	def train(self):
		previous_global_trainer = None
		for task in range(self.task_count):
			# increment task id in prompting modules
			previous_global_trainer = copy.deepcopy(self.global_trainer.learner)
			self.global_trainer.before_task(None, None, task, self.comunication_round_count, 0)
			if task > 0:
				try:
					if self.global_trainer.learner.model.module.prompt is not None:
						self.global_trainer.learner.model.module.prompt.process_task_count()
				except:
					if self.global_trainer.learner.model.prompt is not None:
						self.global_trainer.learner.model.prompt.process_task_count()

			for comunication_round in range(self.comunication_round_count):
				for j in range(self.client_count):
					# torch.cuda.memory._record_memory_history()			   # 开始记录
					# # distribute model to clients	 
					if comunication_round == 0 and task != 0:
						self.learner_pool[j].before_task(self.global_trainer.learner.model, previous_global_trainer.model, task, comunication_round, j)
					else:
						self.learner_pool[j].before_task(self.global_trainer.learner.model, None, task, comunication_round, j)

					# train model
					self.learner_pool[j].train(self.avg_metrics)
					# torch.cuda.memory._dump_snapshot("my_snapshot.pickle")   # 保存文件
					# torch.cuda.memory._record_memory_history(enabled=None)   # 终止记录
					# exit(0)

				# aggregate model
				average_weight = federated_average(self.learner_pool, self.client_weight)
				self.global_trainer.learner.model.load_state_dict(average_weight)
				model_save_dir = self.global_trainer.model_top_dir + f'/models/repeat-{self.seed + 1}/client-{0}/task-{task+1}/comunication_round-{comunication_round}/'
				if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
				self.global_trainer.learner.save_model(model_save_dir)
	
	def evaluate(self):
		for client_index in range(self.client_count):
			self.learner_pool[client_index].evaluate(self.avg_metrics, comunication_round=self.comunication_round_count - 1)
		print(f"-------------------------global--------------------------")
		self.global_trainer.evaluate(self.avg_metrics, comunication_round=self.comunication_round_count - 1)

		print(self.avg_metrics)
	
		write_dict_to_file(self.avg_metrics, self.args.log_dir + f'/matrics/repeat-{self.seed + 1}' + '/avg_metrics.txt')
