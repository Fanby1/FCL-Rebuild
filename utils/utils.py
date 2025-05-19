import sys
import copy
import json
from datetime import datetime

# want to save everything printed to outfile
class Logger(object):
	def __init__(self, name):
		self.terminal = sys.stdout
		self.log = open(name, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)  

	def flush(self):
		self.log.flush()
  
def set_seed(seed):
	"""Set random seed for reproducibility."""
	import random
	import numpy as np
	import torch

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
 
def federated_average(all_client_trainers, num_samples, class_mask = None):
	"""
	Computes the federated average of a list of PyTorch models.
	"""
	w_avg = copy.deepcopy(all_client_trainers[0].learner.model.state_dict()) # create a new model with the same structure as the first client model
	total_weight = sum(num_samples)

	# iterate over the clients and get the class mask
	if class_mask is None:
		class_mask = [] # create a new class mask
		for trainer in all_client_trainers:
			client_class_mask = trainer.learner.class_mask
			class_mask.extend(client_class_mask) # extend the class mask with the client class mask
		class_mask = sorted(list(set(class_mask)))
			
	# create a mask frequency for the classes
	class_frequency = {}
	for trainer in all_client_trainers: # iterate over the clients
		client_class_frequency = trainer.learner.class_frequency # get the class frequency for the client
		for index in client_class_frequency.keys():
			if index in class_frequency:
				class_frequency[index] += client_class_frequency[index]
			else:
				class_frequency[index] = client_class_frequency[index]
	   
	for key in w_avg.keys(): # iterate over the keys of the model
		weighted_sum = None
		for i in range(len(num_samples)): # iterate over the cleint weights
			if key.startswith('last'):
				# aggregate the last layer weights by frequency
				weight = all_client_trainers[i].learner.model.state_dict()[key]
				out_dim = all_client_trainers[i].learner.out_dim
				client_class_frequency = all_client_trainers[i].learner.class_frequency
				client_class_mask = all_client_trainers[i].learner.class_mask
				client_class_mask = generate_index(range(out_dim), client_class_mask).float()
				try:
					weight = client_class_mask * weight
				except:
					weight = client_class_mask.view(-1, 1) * weight
				for class_index in client_class_frequency.keys():
					try:
						weight[class_index, :] *= client_class_frequency[class_index] / class_frequency[class_index]
					except:
						weight[class_index] *= client_class_frequency[class_index] / class_frequency[class_index]
	 
				if weighted_sum is None:
					weighted_sum = weight
				else:
					weighted_sum += weight
			elif key.startswith('prompt'):
				# print(all_client_weights)
				weight = num_samples[i] / total_weight
				if weighted_sum is None:
					weighted_sum = weight * all_client_trainers[i].learner.model.state_dict()[key]
				else:
					weighted_sum += weight * all_client_trainers[i].learner.model.state_dict()[key]
			else:
				pass
		if weighted_sum	== None:	
			weighted_sum = all_client_trainers[0].learner.model.state_dict()[key]
		w_avg[key] = weighted_sum
	return w_avg

import torch

def generate_index(input_list, target_list):
	"""
	根据目标列表生成索引张量，将目标列表中存在的数对应位置置为1，其他位置为0。

	Args:
		input_list (list): 输入列表，表示所有可能的数。
		target_list (list): 目标列表，表示需要置为1的数。

	Returns:
		torch.Tensor: 索引张量，形状与 input_list 相同。
	"""
	# 转换为集合以加速查找
	target_set = set(target_list)

	# 生成索引张量
	index_tensor = torch.tensor([1 if x in target_set else 0 for x in input_list], dtype=torch.int)

	return index_tensor

def trace_handler(prof: torch.profiler.profile):
   # 获取时间用于文件命名
   timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
   file_name = f"visual_mem_{timestamp}"

   # 导出tracing格式的profiling
   prof.export_chrome_trace(f"{file_name}.json")

   # 导出mem消耗可视化数据
   prof.export_memory_timeline(f"{file_name}.html", device="cuda:0")
   
def write_dict_to_file(dictionary, file_path):
    """
    将字典写入文件。

    参数:
        dictionary (dict): 要写入的字典。
        file_path (str): 文件路径。
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(dictionary, file, ensure_ascii=False, indent=4)
