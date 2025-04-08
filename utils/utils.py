import sys
import copy

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
	for key in w_avg.keys(): # iterate over the keys of the model
		weighted_sum = None
		for i in range(len(num_samples)): # iterate over the cleint weights
			# print(all_client_weights)
			weight = num_samples[i] / total_weight
			if weighted_sum is None:
				weighted_sum = weight * all_client_trainers[i].learner.model.state_dict()[key]
			else:
				weighted_sum += weight * all_client_trainers[i].learner.model.state_dict()[key]
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