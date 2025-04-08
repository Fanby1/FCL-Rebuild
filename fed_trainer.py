import dataloaders
import learners
from torch.utils.data import random_split
from trainer import Trainer

class Fed_Trainer:
	def __init__(self, args, seed, metric_keys, save_keys):
		if args.dataset == 'CIFAR100':
			DataSpliter = dataloaders.Cifar100_Spliter
		elif args.dataset == 'ImageNet_R':
			DataSpliter = dataloaders.ImageNetR_Spliter
		else:
			raise ValueError('Dataset not implemented!')
			
		# create spliter
		dataset_spliter = DataSpliter(client_num=5, task_num=5, private_class_num=40, input_size=224, path='C:/Users/Admin/datasets')

		# split dataset
		client_data, client_mask= dataset_spliter.random_split()
		
		client_data_train = []
		client_data_val = []
		for subset in client_data:
			temp_train = []
			temp_val =[]
			for data in subset:
				train_dataset, val_dataset = random_split(data, [int(len(data) * 0.7), len(data) - int(len(data) * 0.7)])
				temp_train.append(train_dataset)
				temp_val.append(val_dataset)
			client_data_train.append(temp_train)
			client_data_val.append(temp_val)
		surro_data, test_data = dataset_spliter.process_testdata(5)
  
		# create trainers
		self.trainers = []
		for trainer_id in args.client_nums:
			self.trainers.append(Trainer(args, seed, metric_keys, save_keys, client_data_train[trainer_id], client_data_val[trainer_id], test_data[trainer_id], client_mask[trainer_id]))
		
		# create server model
		
		# model load directory
		self.model_top_dir = args.log_dir

		# select dataset
		self.grayscale_vis = False
		self.top_k = 1
		if args.dataset == 'CIFAR100':
			num_classes = 100
			self.dataset_size = [32,32,3]
		elif args.dataset == 'ImageNet_R':
			num_classes = 200
			self.dataset_size = [224,224,3]
			self.top_k = 1
		else:
			raise ValueError('Dataset not implemented!')

		# upper bound flag
		if args.upper_bound_flag:
			args.other_split_size = num_classes
			args.first_split_size = num_classes

		# load tasks
		self.num_tasks = len(client_mask[0])
		self.task_logist = None
		self.task_names = [str(i+1) for i in range(self.num_tasks)]

		# Prepare the self.learner (model)
		self.learner_config = {'num_classes': num_classes,
						'lr': args.lr,
						'debug_mode': args.debug_mode == 1,
						'momentum': args.momentum,
						'weight_decay': args.weight_decay,
						'schedule': args.schedule,
						'schedule_type': args.schedule_type,
						'model_type': args.model_type,
						'model_name': args.model_name,
						'optimizer': args.optimizer,
						'gpuid': args.gpuid,
						'memory': args.memory,
						'temp': args.temp,
						'out_dim': num_classes,
						'overwrite': args.overwrite == 1,
						'DW': args.DW,
						'batch_size': args.batch_size,
						'upper_bound_flag': args.upper_bound_flag,
						'tasks': self.tasks_logits,
						'top_k': self.top_k,
						'prompt_param':[self.num_tasks,args.prompt_param]
						}
		self.learner_type, self.learner_name = args.learner_type, args.learner_name
		self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
	def train(self, avg_metrics):
		for trainer in self.trainers:
			trainer.train(avg_metrics)
		