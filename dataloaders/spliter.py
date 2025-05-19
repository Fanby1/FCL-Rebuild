import random
import numpy as np
import torch

from tqdm import tqdm
from typing import Sequence

from dataloaders.utils import build_transform
from dataloaders.customed_dataset import CustomedSubset
from torchvision.transforms import transforms

from dataloaders.continual_datasets import Imagenet_R
from torchvision.datasets import MNIST,CIFAR100

class Spliter():
    
	class_num: int
	train_set: object
	test_set: object
	train_index_by_class_label: list[set]
	train_class_counts: list[int]
	test_index_by_class_label: list[set]
	test_class_counts: list[int]
	public_class_num: int
	private_class_num: int
	dirichlet_perclass: dict[int, float]
	client_mask: list[list[list[int]]]
	class_public: list[int]
	class_private: list[list[int]]

	def __init__(self,client_num, attacker_num, task_num, private_class_num, input_size, path):
		self.client_num = client_num
		self.attacker_num = attacker_num
		self.attacker_client_index = [i for i in range(client_num)]
		self.attacker_client_index = random.sample(self.attacker_client_index, attacker_num)
		self.task_num = task_num

		self.private_class_num = private_class_num
		self.input_size = input_size
		
		self.create_dataset(path)

	def create_dataset(self):
		raise NotImplementedError("create_dataset method not implemented")

	def statistic_dataset_by_label(self, data_set: Sequence[tuple[int, int]]):
		print("statistic dataset by label")
		# 统计每个类的数量和index
		class_counts = [0] * self.class_num #每个类的数量
		index_by_class_label = [set() for _ in range(self.class_num)] # 每个类的index
   
		for idx, (_, label) in enumerate(tqdm(data_set)):
			class_counts[label] += 1
			index_by_class_label[label].add(idx)

		# index_by_class_label 里保存了每个类的index
		return class_counts, index_by_class_label

	def sample_k_indice(self, indices: set[int], k: int) -> list[int]:
		# 随机选择k个index
		random_index = random.sample(indices, k=k)
		for i in random_index:
			indices.remove(i)
		return random_index

	def random_split_class_mask(self):
		self.client_mask = [[] for _ in range(0,self.client_num + self.attacker_num)]

		# 分类
		total_private_class_num = self.client_num * self.private_class_num
		self.public_class_num = self.class_num - total_private_class_num
		class_public = set(range(self.class_num))

		class_p = random.sample(class_public, total_private_class_num)
		self.class_public = list(set(class_public) - set(class_p))
		print(f"public class: {self.class_public}")
		self.class_every_task = (self.public_class_num + self.private_class_num) // self.task_num

		self.class_private = [None] * self.client_num
		for client_index in tqdm(range(self.client_num)):
			self.class_private[client_index] = class_p[self.private_class_num * client_index : self.private_class_num * (client_index + 1)]
			self.class_private[client_index].extend(self.class_public)
			random.shuffle(self.class_private[client_index])
			self.client_mask[client_index] = self.random_split_class_in_client(client_index)
			for attacker_client_index in range(self.attacker_num):
				if client_index == self.attacker_client_index[attacker_client_index]:
					self.client_mask[attacker_client_index + self.client_num] = self.client_mask[client_index]
		return self.client_mask, self.class_public, self.class_private

	def random_split_class_in_client(self, client_index: int) -> list[list[int]]:
		class_mask = []
		for task_index in range(self.task_num):
			class_this_task = self.random_split_class_in_task(client_index, task_index)
			class_mask.append(class_this_task)
		return class_mask

	def random_split_class_in_task(self, client_index: int, task_index: int) -> list[int]:
		class_private = self.class_private[client_index]
		client_mask = class_private[task_index * self.class_every_task: (task_index + 1) * self.class_every_task]
		return client_mask
		
	def random_split_subsets(self, data_set: Sequence[tuple[int, object]], train: bool, class_counts: list[int], 
                          index_by_class_label: list[set[int]]) -> tuple[list[CustomedSubset], list[list[int]]]:
		trans = build_transform(train,self.input_size)

		client_subset = [[None] * self.task_num for _ in range(0,self.client_num + self.attacker_num)]
  
		self.dirichlet_perclass = [None] * self.class_num
		for class_index in self.class_public:
			a = np.random.dirichlet(np.ones(self.client_num), 1)
			while  (a < (1 / self.client_num / 2)).any():
				a = np.random.dirichlet(np.ones(self.client_num), 1)
			self.dirichlet_perclass[class_index] = a[0]
   
		for client_index in tqdm(range(self.client_num)):
			for task_index in range(self.task_num):
				sample_indices = self.random_split_indices_in_task(client_index, task_index,
                                    class_counts, index_by_class_label)
				client_subset[client_index][task_index] = CustomedSubset(data_set, sample_indices, trans, None)
				for attacker_index in range(self.attacker_num):
					if client_index == self.attacker_client_index[attacker_index]:
						client_subset[attacker_index + self.client_num][task_index] = CustomedSubset(data_set, sample_indices, trans, None, attacker=True)
		return client_subset
		

	def random_split_indices_in_task(self, client_index: int, task_index: int,
                                class_counts: list[int], index_by_class_label: list[set[int]]) -> list[int]:
		class_mask = self.client_mask[client_index][task_index]	
		# print(f"client_index: {client_index}, task_index: {task_index}, class_mask: {class_mask}")
		result_index = []
		for class_index in class_mask:
			if class_index in self.class_public:
				# 是公共类
				length = int(class_counts[class_index] * self.dirichlet_perclass[class_index][client_index])
				random_index = self.sample_k_indice(index_by_class_label[class_index], length)
				result_index.extend(random_index)
			else: #是私有类
				random_index = list(index_by_class_label[class_index])
				result_index.extend(random_index)
		random.shuffle(result_index)
		return result_index


	# 分成client_num数目个subset,每个subset里包含了task个subsubset
	def random_split(self):
		client_mask, class_public, class_private = self.random_split_class_mask()
		client_subset = self.random_split_subsets(self.train_set, True, self.train_class_counts.copy(), 
                                            self.train_index_by_class_label.copy())

		return client_subset,client_mask


	def process_testdata(self,surrogate_num):
		trans = build_transform(False,self.input_size)
		class_counts = self.test_class_counts
		index_by_class_label = self.test_index_by_class_label.copy()

		surro_index =[]
		test_index = []
		for class_index in tqdm(range(self.class_num)):
			q = 0
			unused_indice = index_by_class_label[class_index]

			while q < surrogate_num:
				random_index = random.choice(list(unused_indice))
				surro_index.append(random_index)
				unused_indice.remove(random_index)
				q += 1
			test_index.extend(list(unused_indice))
		surrodata = CustomedSubset(self.test_set,surro_index,trans,None)
		testdata = CustomedSubset(self.test_set,test_index,trans,None)
		return surrodata,testdata

	def random_split_synchron(self):
		trans = build_transform(True,self.input_size)
		# self.Imagenet_R = Imagenet_R(root='C:/Users/Admin/datasets', train=True, download=True)
		trainset = self.train_set

		# 100个类别的数据分给三个客户端使用
		class_counts = self.train_class_counts.copy() # 每个类的数量
		class_label = self.train_index_by_class_label.copy() # 每个类的index

		class_public = [i for i in range(200)]
		# 对每个客户端进行操作
		client_subset = [[] for i in range(0,self.client_num)]
		client_mask = [[] for i in range(0,self.client_num)]

		class_every_task = 10
		dirichlet_perclass = {}

		for i in range(0,self.client_num):
			for j in range(0,self.task_num):
				index = []
				class_this_task = class_public[j*class_every_task: j*class_every_task+class_every_task]
				client_mask[i].append(class_this_task)

				# 是公共类
				for k in class_this_task:
					length = int(int(class_counts[k]) * random.uniform(0.3, 0.85))
					unused_indice = set(class_label[k])
					index.extend(self.sample_k_indice(unused_indice, length))

				random.shuffle(index)
				client_subset[i].append(CustomedSubset(trainset, index, trans,None))

		return client_subset,client_mask

class ImageNetR_Spliter(Spliter):
	def __init__(self, client_num, attacker_num, task_num, private_class_num, input_size, path):
		super().__init__(client_num, attacker_num, task_num, private_class_num, input_size, path)
  
	def create_dataset(self, path):
		Imagenet_R = Imagenet_R(root=path, train=True, download=True)
		Imagenet_R_test = Imagenet_R(root=path, train=False, download=True)

		self.class_num = 200
		self.train_set = Imagenet_R
		self.test_set = Imagenet_R_test

		self.train_class_counts, self.train_index_by_class_label = self.statistic_dataset_by_label(self.train_set)
		self.test_class_counts, self.test_index_by_class_label = self.statistic_dataset_by_label(self.test_set)
  
class Cifar100_Spliter(Spliter):
	def __init__(self, client_num, attacker_num, task_num, private_class_num, input_size, path):
		super().__init__(client_num, attacker_num, task_num, private_class_num, input_size, path)
  
	def create_dataset(self, path):
		cifar100_dataset = CIFAR100(root=path, train=True, download=True)
		cifar100_test_dataset = CIFAR100(root=path, train=False, download=True)

		self.class_num = 100
		self.train_set = cifar100_dataset
		self.test_set = cifar100_test_dataset

		self.train_class_counts, self.train_index_by_class_label = self.statistic_dataset_by_label(self.train_set)
		self.test_class_counts, self.test_index_by_class_label = self.statistic_dataset_by_label(self.test_set)