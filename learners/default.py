from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
import copy
from utils.schedulers import CosineSchedule
import time
class NormalNN(nn.Module):
	'''
	Normal Neural Network with SGD for classification
	'''
	def __init__(self, learner_config):

		super(NormalNN, self).__init__()
		self.log = print
		self.config = learner_config
		self.out_dim = learner_config['out_dim']
		self.model = self.create_model()
		self.reset_optimizer = True
		self.overwrite = learner_config['overwrite']
		self.batch_size = learner_config['batch_size']
		self.tasks = learner_config['tasks']
		self.top_k = learner_config['top_k']

		# replay memory parameters
		self.memory_size = self.config['memory']
		self.task_count = 0
  
		# count trained classes
		self.class_mask = []
		self.class_frequency = {}

		# class balancing
		self.dw = self.config['DW']
		if self.memory_size <= 0:
			self.dw = False
   
		# check if cuda is available
		if self.config['gpuid'][0] >= 0:
			self.gpu = True

		# supervised criterion
		self.criterion_fn = nn.CrossEntropyLoss(reduction='none')

		# set up schedules
		self.schedule_type = self.config['schedule_type']
		self.schedule = self.config['schedule']

		# initialize optimizer
		self.init_optimizer()
  
		# previous global model
		self.global_model = None
		self.previous_task_model = None

	##########################################
	#		   MODEL TRAINING			   #
	##########################################

	def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None, task_index=-1, class_mask=None):
		print('Train...')
		print(class_mask)
		# try to load model
		need_train = True
		if not self.overwrite:
			try:
				self.load_model(model_save_dir)
				need_train = False
			except:
				self.cuda()
				pass

		# update class mask
		self.class_mask.extend(class_mask)
		self.class_mask = sorted(list(set(self.class_mask)))
		self.curr_class_mask = sorted(class_mask)
		for class_index in class_mask:
			if class_index in self.class_frequency:
				self.class_frequency[class_index] += 1
			else:
				self.class_frequency[class_index] = 1
  
		# trains
		if self.reset_optimizer and self.conmunication_round == 0:  # Reset optimizer before learning each task
			self.log('Optimizer is reset!')
			self.init_optimizer()
		if need_train:
			# print('-----------------------------begin training-----------------------------')
			# print(torch.cuda.memory_summary())
			# time.sleep(10000)
			# data weighting
			self.data_weighting(train_dataset)
			losses = AverageMeter()
			acc = AverageMeter()
			batch_time = AverageMeter()
			batch_timer = Timer()
			for epoch in range(self.config['schedule'][0]):
				self.epoch=epoch
				if epoch > 0: self.scheduler.step()
				for param_group in self.optimizer.param_groups:
					self.log('LR:', param_group['lr'])
				batch_timer.tic()
				for i, (x, y)  in enumerate(train_loader):
					
				
					task = task_index
					# verify in train mode
					self.model.train()

					# send data to gpu
					if self.gpu:
						x = x.cuda()
						y = y.cuda()
					
					# model update
					loss, output= self.update_model(x, y)

					# measure elapsed time
					batch_time.update(batch_timer.toc())  
					batch_timer.tic()
					
					# measure accuracy and record loss
					y = y.detach()
					accumulate_acc(output, y, task, acc, topk=(self.top_k,))
					losses.update(loss,  y.size(0)) 
					batch_timer.tic()

				# eval update
				self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][0]))
				self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))

				# reset
				losses = AverageMeter()
				acc = AverageMeter()
				

		self.model.eval()
		self.cpu()

		try:
			return batch_time.avg
		except:
			return None

	def criterion(self, logits, targets, data_weights):
		loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
		return loss_supervised 

	def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
		
		dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()] 
		logits = self.forward(inputs) 
  
		# remap targets to class mask
		for i in range(len(self.class_mask)):
			targets[targets == self.class_mask[i]] = i

		total_loss = self.criterion(logits, targets.long(), dw_cls)

		self.optimizer.zero_grad()
		total_loss.backward()
		self.optimizer.step()
		return total_loss.detach(), logits

	def after_task(self, train_dataset):
		# Extend memory
		if self.memory_size > 0:
			train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))
   
	def before_task(self, global_model, previous_task_global, task_index, comunication_round, client_index):
		print(f'update client {client_index} global model...')

		# distribute model to client
		try :
			self.model.load_state_dict(copy.deepcopy(global_model.state_dict()))
		except:
			pass
  
		# set task id for model (needed for fed)
		self.curr_task = task_index
		self.model.task_count = task_index
		self.model.prompt.task_count = task_index
		self.model.task_id = task_index
		self.conmunication_round = comunication_round
		self.client_index = client_index
  
		# save previous model
		if previous_task_global is not None:
			self.previous_task_model = copy.deepcopy(previous_task_global)
		self.global_model = copy.deepcopy(global_model)

	def validation(self, dataloader, model=None, class_mask = None, task_metric='acc',  verbal = True, task_global=False, task_index=-1):
		print('Validation...')
		print(class_mask)
		torch.cuda.empty_cache()  # 清空缓存
		if model is None:
			model = self.model

		if self.gpu:
			model.cuda()
		# This function doesn't distinguish tasks.
		batch_timer = Timer()
		acc = AverageMeter()
		batch_timer.tic()

		orig_mode = model.training
		model.eval()
		for i, (input, target) in enumerate(dataloader):
			task = task_index
			if self.gpu:
				with torch.no_grad():
					input = input.cuda()
					target = target.cuda()
			
			# # 构造 mask，筛选出 target 中属于 class_mask 的样本
			# mask = torch.zeros_like(target, dtype=torch.bool)  # 初始化全为 False 的布尔掩码
			# for cls in class_mask:
			# 	mask |= (target == cls)  # 将属于 class_mask 的类别置为 True
	
			# # 获取有效样本的索引
			# mask_ind = mask.nonzero(as_tuple=False).view(-1)
			# input, target = input[mask_ind], target[mask_ind]
			
			if len(target) > 1:
				output = model.forward(input)[:, class_mask]
					
				# remap targets to class mask
				for i in range(len(class_mask)):
					target[target == class_mask[i]] = i
	
				acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
	
		model.train(orig_mode)

		if verbal:
			self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
					.format(acc=acc, time=batch_timer.toc()))
   
		model.cpu()
		return acc.avg

	##########################################
	#				MODEL UTILS				 #
	##########################################

	# data weighting
	def data_weighting(self, dataset, num_seen=None):
		self.dw_k = torch.tensor(np.ones(len(self.class_mask) + 1, dtype=np.float32))
		# cuda
		if self.cuda:
			self.dw_k = self.dw_k.cuda()

	def save_model(self, filename):
		model_state = self.model.state_dict()
		for key in model_state.keys():  # Always save it to cpu
			model_state[key] = model_state[key].cpu()
		self.cpu()
  
		# 检查目标文件夹是否已有内容
		self.log('=> Saving class model to:', filename)
		folder = os.path.dirname(filename)
		if os.path.exists(folder) and len(os.listdir(folder)) > 0:
			self.log(f"Folder '{folder}' is not empty. Skipping save.")
			return
		torch.save(model_state, filename + 'class.pth')
		self.log('=> Save Done')
		

	def load_model(self, filename):
		self.model.load_state_dict(torch.load(filename + 'class.pth'))
		self.log('=> Load Done')
		self.cuda()
		self.model.eval()

	def load_model_other(self, filename, model):
		model.load_state_dict(torch.load(filename + 'class.pth'))
		model = model.cuda()
		return model.eval()

	# sets model optimizers
	def init_optimizer(self):

		# parse optimizer args
		optimizer_arg = {'params':self.model.parameters(),
						 'lr':self.config['lr'],
						 'weight_decay':self.config['weight_decay']}
		if self.config['optimizer'] in ['SGD','RMSprop']:
			optimizer_arg['momentum'] = self.config['momentum']
		elif self.config['optimizer'] in ['Rprop']:
			optimizer_arg.pop('weight_decay')
		elif self.config['optimizer'] == 'amsgrad':
			optimizer_arg['amsgrad'] = True
			self.config['optimizer'] = 'Adam'
		elif self.config['optimizer'] == 'Adam':
			optimizer_arg['betas'] = (self.config['momentum'],0.999)

		# create optimizers
		self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
		
		# create schedules
		if self.schedule_type == 'cosine':
			self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[0] * self.schedule[1])
		elif self.schedule_type == 'decay':
			self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule[0] * self.schedule[1], gamma=0.1)

	def create_model(self):
		cfg = self.config

		# Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
		model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim)

		return model

	def print_model(self):
		self.log(self.model)
		self.log('#parameter of model:', self.count_parameter())
	
	def reset_model(self):
		self.model.apply(weight_reset)

	def forward(self, x):
		# forward pass
		return self.model.forward(x)[:, self.class_mask]

	def predict(self, inputs):
		self.model.eval()
		out = self.forward(inputs)
		return out

	def count_parameter(self):
		return sum(p.numel() for p in self.model.parameters())   

	def count_memory(self, dataset_size):
		return self.count_parameter() + self.memory_size * dataset_size[0]*dataset_size[1]*dataset_size[2]

	def cuda(self):
		if not self.gpu:
			self.log('No GPU available')
			return self
		torch.cuda.set_device(self.config['gpuid'][0])
		self.model = self.model.cuda()
		self.criterion_fn = self.criterion_fn.cuda()
  
		try:
			self.global_model.cuda()
		except:
			self.global_model = None

		try:
			self.previous_task_model.cuda()
		except:
			self.previous_task_model = None

		# Multi-GPU
		if len(self.config['gpuid']) > 1:
			self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
		return self

	def cpu(self):
		"""将模型和损失函数从 GPU 移动到 CPU"""
		self.model = self.model.cpu() 
		self.criterion_fn = self.criterion_fn.cpu()	
  
		try:
			self.global_model.cpu()
		except:
			pass

		try:
			self.previous_task_model.cpu()
		except:
			pass
  
		if isinstance(self.model, torch.nn.DataParallel):
			self.model = self.model.module.cpu()	
		return self

	def _get_device(self):
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.log("Running on:", device)
		return device

	def pre_steps(self):
		pass

class FinetunePlus(NormalNN):

	def __init__(self, learner_config):
		super(FinetunePlus, self).__init__(learner_config)

	def update_model(self, inputs, targets, target_KD = None):

		# get output
		logits = self.forward(inputs)

		# standard ce
		logits[:,:self.last_valid_out_dim] = -float('inf')
		dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
		total_loss = self.criterion(logits, targets.long(), dw_cls)

		self.optimizer.zero_grad()
		total_loss.backward()
		self.optimizer.step()
		return total_loss.detach(), logits

def weight_reset(m):
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
		m.reset_parameters()

def accumulate_acc(output, target, task, meter, topk):
	meter.update(accuracy(output, target, topk), len(target))
	return meter