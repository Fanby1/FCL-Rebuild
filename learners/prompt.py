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
from .default import NormalNN, weight_reset, accumulate_acc
import copy
import torchvision
from utils.schedulers import CosineSchedule
from torch.autograd import Variable, Function
from utils.utils import trace_handler
from torch.autograd.profiler import record_function

class TripletLoss(torch.nn.Module):
	def __init__(self, margin=1.0):
		super(TripletLoss, self).__init__()
		self.margin = margin

	def forward(self, anchor, positive, negative):
		distance_positive = (anchor - positive).pow(2).sum(1) * 1.0e3  # scale up
		distance_negative = (anchor - negative).pow(2).sum(1) * 1.0e1  # scale up
		# print(distance_positive)
		losses = torch.relu((distance_positive - distance_negative).sum() + self.margin)
		return losses.mean()
class PositiveLoss(torch.nn.Module):
	def __init__(self, margin=0.0):
		super(PositiveLoss, self).__init__()
		self.margin = margin

	def forward(self, anchor, positive):
		distance_positive = (anchor - positive).pow(2).sum(1) * 1.0e3 # scale up
		losses = torch.relu((distance_positive).sum() + self.margin)
		return losses.mean()

class Prompt(NormalNN):

	def __init__(self, learner_config):
		self.prompt_param = learner_config['prompt_param']
		super(Prompt, self).__init__(learner_config)

	def update_model(self, inputs, targets):

		# logits
		logits, prompt_loss, _ = self.model(inputs, train=True)
		logits = logits[:,self.class_mask]
	
		for i in range(len(self.class_mask)):
			targets[targets == self.class_mask[i]] = i

		# ce with heuristic
		for i in range(len(self.class_mask)):
			if self.class_mask[i] not in self.curr_class_mask:
				logits[:,i] = -float('inf')
    
		# # logists
		# logits, prompt_loss, _ = self.model(inputs, train=True) # original model
		# logits = logits[:,self.curr_class_mask]
		
		# # targets
		# for i in range(len(self.curr_class_mask)):
		# 	targets[targets == self.curr_class_mask[i]] = i

		# ce with heuristic
		dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
		# dw_cls *= 0.5
		total_loss = self.criterion(logits, targets.long(), dw_cls)

		# ce loss
		total_loss = total_loss + prompt_loss.sum()

		# step
		self.optimizer.zero_grad()
		total_loss.backward()
		self.optimizer.step()

		return total_loss.detach(), logits

	# sets model optimizers
	def init_optimizer(self):

		# parse optimizer args
		# Multi-GPU
		if len(self.config['gpuid']) > 1:
			params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
		else:
			params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())
		print('*****************************************')
		optimizer_arg = {'params':params_to_opt,
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
		pass

	def cuda(self):
		super(Prompt, self).cuda()
		return self

	def cpu(self):
		"""将模型和损失函数从 GPU 移动到 CPU"""
		super(Prompt, self).cpu()
		return self

# Our method!
class CODAPrompt(Prompt):

	def __init__(self, learner_config):
		super(CODAPrompt, self).__init__(learner_config)

	def create_model(self):
		cfg = self.config
		model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'coda',prompt_param=self.prompt_param)
		return model

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(Prompt):

	def __init__(self, learner_config):
		super(DualPrompt, self).__init__(learner_config)

	def create_model(self):
		cfg = self.config
		model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'dual', prompt_param=self.prompt_param)
		return model

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(Prompt):

	def __init__(self, learner_config):
		super(L2P, self).__init__(learner_config)

	def create_model(self):
		cfg = self.config
		model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'l2p',prompt_param=self.prompt_param)
		return model

class CPrompt(Prompt):

	def __init__(self, learner_config):
		self.prompt_param = learner_config['prompt_param']
		super(CPrompt, self).__init__(learner_config)
		self.triplet_loss = TripletLoss(margin=1)
		self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
  
		self.fedmoon = learner_config['fedmoon']
		self.tau = learner_config['tau']
  
	def create_model(self):
		cfg = self.config
		model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'cprompt',prompt_param=self.prompt_param)
		return copy.deepcopy(model)

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
			# data weighting
			self.data_weighting(train_dataset)
			losses = [AverageMeter() for _ in range(4)]
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
					loss, loss_class, loss_distill, output, fedmoonLoss = self.update_model(x, y)

					# measure elapsed time
					batch_time.update(batch_timer.toc())  
					batch_timer.tic()
					
					# measure accuracy and record loss
					y = y.detach()
					accumulate_acc(output, y, task, acc, topk=(self.top_k,))
					losses[0].update(loss,  y.size(0)) 
					losses[1].update(loss_class,  y.size(0)) 
					losses[2].update(loss_distill,  y.size(0))
					losses[3].update(fedmoonLoss,  y.size(0))
					batch_timer.tic()

				# eval update
				self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][0]))
				self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f} | C2 loss {lossp.avg:.3f}'.format(loss=losses[0],acc=acc,lossp=losses[3]))
            
				# reset
				losses = [AverageMeter() for _ in range(4)]
				acc = AverageMeter()
				
		self.model.eval()

		try:
			return batch_time.avg
		except:
			return None

	# update model - add dual prompt loss   
	def update_model(self, inputs, targets):
		fedmoonLoss = torch.zeros((1,), requires_grad=True).cuda()
		self.optimizer.zero_grad()
		t_c2loss = torch.zeros((1,), requires_grad=True).cuda()
		
		# logits
		logits, prompt_loss, prelogits_current = self.model(inputs, train=True) # original model

		if self.fedmoon[0] == 2:
			with torch.no_grad(): 
				_, _, prelogits_global = self.global_model(inputs, train=True)
			posi = self.cos(prelogits_current, prelogits_global)
			if self.previous_task_model is not None:
				with torch.no_grad():
					_, _, prelogits_previous = self.previous_task_model(inputs, train=True)
				nega = self.cos(prelogits_current, prelogits_previous)

				numerator = torch.exp(posi/self.tau)
				denominator = torch.exp(posi/self.tau) + torch.exp(nega/self.tau)
				fedmoonLoss = -torch.log(numerator/denominator)

				fedmoonLoss = fedmoonLoss.mean()
		elif self.fedmoon[0] == 1:
			with torch.no_grad(): 
				_, _, prelogits_global = self.global_model(inputs, train=True)
			posi = self.cos(prelogits_current, prelogits_global) # decrease the distance between current task and global model
			if self.previous_task_model is not None:
				with torch.no_grad():
					_, _, prelogits_previous_task = self.previous_task_model(inputs, train=True)
				nega = self.cos(prelogits_current, prelogits_previous_task) # increase the distance between current task and previous task model
				numerator = torch.exp(posi/self.tau)
				denominator = torch.exp(posi/self.tau) + torch.exp(nega/self.tau)
				fedmoonLoss = -torch.log(numerator/denominator)
				fedmoonLoss = fedmoonLoss.mean()
		elif self.fedmoon[0] == 3:				
			if self.previous_task_model is not None:
				with torch.no_grad():
					_, _, prelogits_previous_task = self.previous_task_model(inputs, train=True)
					_, _, prelogits_global = self.global_model(inputs, train=True)
				_, _, prelogits_previous_task = self.previous_task_model(inputs, train=True)
				t_c2loss =self.triplet_loss(prelogits_current, prelogits_global, prelogits_previous_task)
			if self.previous_task_model is None:
				with torch.no_grad():
					_, _, prelogits_global = self.global_model(inputs, train=True)
				t_c2loss = PositiveLoss()(prelogits_current, prelogits_global)
  
		# logists
		logits = logits[:,self.class_mask]

		# targets
		for i in range(len(self.class_mask)):
			targets[targets == self.class_mask[i]] = i

		# ce with heuristic
		for i in range(len(self.class_mask)):
			if self.class_mask[i] not in self.curr_class_mask:
				logits[:,i] = -float('inf')
		dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
		# dw_cls *= 0.5
		total_loss = self.criterion(logits, targets.long(), dw_cls)

		total_loss = total_loss  +  t_c2loss + prompt_loss # mu is the self.muMoon * +  self.muMoon * fedmoonLoss

		self.optimizer.zero_grad()
		total_loss.backward()
		self.optimizer.step()
		fedmoonLoss = t_c2loss

		return total_loss.detach(), prompt_loss.sum().detach(), torch.zeros((1,), requires_grad=True).cuda().detach(), logits , fedmoonLoss.detach()