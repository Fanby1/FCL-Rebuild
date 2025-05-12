import os
import sys
import argparse
import torch
import numpy as np
import random
from random import shuffle
from collections import OrderedDict
import dataloaders
from dataloaders.utils import *
from torch.utils.data import DataLoader
import learners

class Trainer:

	def __init__(self, args, seed, metric_keys, save_kays, train_dataset=None, test_dataset=None, validate_dataset=None, class_mask=None, client_index=-1):

		# process inputs
		self.seed = seed
		self.metric_keys = metric_keys
		self.save_keys = save_kays
		self.log_dir = args.log_dir
		self.batch_size = args.batch_size
		self.workers = args.workers
		self.client_index = client_index
		self.curr_task = 0
		self.conmunication_round = 0
		
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
		self.train_dataset = train_dataset
		self.test_dataset = test_dataset
		self.validate_dataset = validate_dataset
		for i in range(len(class_mask)):
			class_mask[i] = sorted(class_mask[i])
		self.class_mask = class_mask

		# upper bound flag
		if args.upper_bound_flag:
			args.other_split_size = num_classes
			args.first_split_size = num_classes

		# load tasks
		self.tasks = class_mask
		self.tasks_logits = class_mask
		self.num_tasks = len(self.tasks)
		self.task_names = [str(i+1) for i in range(self.num_tasks)]

		# number of tasks to perform
		if args.max_task > 0:
			self.max_task = min(args.max_task, len(self.task_names))
		else:
			self.max_task = len(self.task_names)

		
		# for oracle
		self.oracle_flag = args.oracle_flag
		self.add_dim = 0

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
						'tau': args.tau,
						'DW': args.DW,
						'batch_size': args.batch_size,
						'upper_bound_flag': args.upper_bound_flag,
						'tasks': self.tasks_logits,
						'top_k': self.top_k,
						'fedmoon': args.fedMoon,
						'prompt_param':[self.num_tasks,args.prompt_param]
						}
		self.learner_type, self.learner_name = args.learner_type, args.learner_name
		self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

	def task_eval(self, t_index, t_last=-1, task='acc', test=False, in_task=False):

		val_name = self.task_names[t_index]
		print('validation split name:', val_name)
  
		if test:
			val_dataset = self.test_dataset[t_index]
		else:
			val_dataset = self.validate_dataset[t_index]
		# eval
		class_mask = []
		if in_task:
			class_mask = self.class_mask[t_index]
		else:
			for i in range(t_last + 1):
				class_mask.extend(self.class_mask[i])
		class_mask = sorted(list(set(class_mask)))
		# val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
		val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2, shuffle=False, drop_last=False, num_workers=self.workers)
		return self.learner.validation(val_loader, class_mask = class_mask, task_metric=task, task_index = t_index)

	def train(self, avg_metrics):
	
		# temporary results saving
		temp_table = {}
		for mkey in self.metric_keys: temp_table[mkey] = []
		temp_dir = self.log_dir + '/temp/'
		if not os.path.exists(temp_dir): os.makedirs(temp_dir)

		# for each task
		
		# save current task index
		self.current_t_index = self.curr_task
  
		# print name
		train_name = self.task_names[self.curr_task]
		print('======================', train_name, '=======================')

		# load dataset for task
		if self.oracle_flag:
			train_dataset = self.train_dataset[self.curr_task]
			self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
		else:
			train_dataset = self.train_dataset[self.curr_task]

		# set task id for model (needed for prompting)
		try:
			self.learner.model.module.task_id = self.curr_task
		except:
			self.learner.model.task_id = self.curr_task

		# load dataset with memory
		# self.train_dataset.append_coreset(only=False)

		# load dataloader
		train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))

		# learn
		val_dataset = self.validate_dataset[self.curr_task]
		val_loader  = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
		model_save_dir = self.model_top_dir + f'/models/repeat-{self.seed+1}/client-{self.client_index}/task-{train_name}/comunication_round-{self.conmunication_round}/'
		if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
		avg_train_time = self.learner.learn_batch(train_loader, train_dataset, model_save_dir, val_loader, task_index = self.curr_task, class_mask = self.class_mask[self.curr_task])

		# evaluate acc
		train_acc = []
		train_acc.append(self.task_eval(self.curr_task, t_last=self.curr_task))
		train_acc.append(self.task_eval(self.curr_task, t_last=self.curr_task, in_task=True))
		train_acc.append(self.task_eval(0, t_last=self.curr_task))
		train_acc.append(self.task_eval(0, t_last=self.curr_task, in_task=True))

		# save temporary acc results
		for mkey in ['acc']:
			save_file = temp_dir + mkey + '.csv'
			np.savetxt(save_file, train_acc, delimiter=",", fmt='%.2f')  

		if avg_train_time is not None: avg_metrics['time'][f'client-{self.client_index}'][self.curr_task] = avg_train_time
		
		# save model
		self.learner.save_model(model_save_dir)
  
		return avg_metrics 

	def evaluate(self, avg_metrics, comunication_round=0):

		self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

		# store results
		metric_table = {}
		metric_table_local = {}
		for mkey in self.metric_keys:
			metric_table[mkey] = {}
			metric_table_local[mkey] = {}
			
		for i in range(self.max_task):

			# increment task id in prompting modules
			if i > 0:
				try:
					if self.learner.model.module.prompt is not None:
						self.learner.model.module.prompt.task_count = i
				except:
					if self.learner.model.prompt is not None:
						self.learner.model.prompt.task_count = i

			# load model
			model_save_dir = self.model_top_dir + f'/models/repeat-{self.seed+1}/client-{self.client_index}/task-{self.task_names[i]}/comunication_round-{comunication_round}/'
			self.learner.task_count = i 
			self.learner.pre_steps()
			self.learner.load_model(model_save_dir)


			# set task id for model (needed for prompting)
			try:
				self.learner.model.module.task_id = i
			except:
				self.learner.model.task_id = i

			# evaluate acc
			skey = 'global' if self.client_index == 0 else f'client-{self.client_index}'
			mkey = 'task-1-acc'
			avg_metrics[mkey][skey][i] = self.task_eval(0, t_last=i, test=False)
   
			mkey = 'last-task-acc'
			avg_metrics[mkey][skey][i] = self.task_eval(i, t_last=i, test=False)
			
			self.learner.cpu()
		
		return avg_metrics

	def before_task(self, model, previous_model, task_index, comunication_round, client_index):
		self.curr_task = task_index
		self.task_id = task_index
		self.conmunication_round = comunication_round
		self.learner.before_task(model, previous_model, task_index, comunication_round, client_index)