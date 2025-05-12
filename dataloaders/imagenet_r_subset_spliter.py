import random
import numpy as np
import torch

from tqdm import tqdm

from dataloaders.continual_datasets import Imagenet_R
from dataloaders.utils import build_transform
from dataloaders.customed_dataset import CustomedSubset


class ImageNetR_Spliter():

	def __init__(self,client_num, attacker_num, task_num, private_class_num, input_size, path):
		self.client_num = client_num
		self.attacker_num = attacker_num
		self.attacker_client_index = [i for i in range(client_num)]
		self.attacker_client_index = random.sample(self.attacker_client_index, attacker_num)
		self.task_num = task_num

		self.private_class_num = private_class_num
		self.input_size = input_size
		self.Imagenet_R = Imagenet_R(root=path, train=True, download=True)
		self.Imagenet_R_test = Imagenet_R(root=path, train=False, download=True)



	# 分成client_num数目个subset,每个subset里包含了task个subsubset
	def random_split(self):
		trans = build_transform(True,self.input_size)
		trainset = self.Imagenet_R

		# 100个类别的数据分给三个客户端使用
		class_counts = torch.zeros(200) #每个类的数量
		class_label = [] # 每个类的index
		for i in range(200):
			class_label.append([])
		j = 0
   
		for idx, (_, label) in enumerate(tqdm(trainset)):
			class_counts[label] += 1
			class_label[label].append(idx)

		# class_label 里保存了每个类的index

		# 分类
		total_private_class_num = self.client_num*self.private_class_num
		public_class_num = 200-total_private_class_num
		class_public = [i for i in range(200)]
		class_public = set(class_public)
		class_public = list(class_public)

		class_p = random.sample(class_public, total_private_class_num)
		class_public = list(set(class_public) - set(class_p))

		class_private = [class_p[self.private_class_num*i : self.private_class_num*i+self.private_class_num] for i in range(0,self.client_num)]
			# random.shuffle(class_private[i])
		# print(class_private)


		# 对每个客户端进行操作
		client_subset = [[] for _ in range(0,self.client_num + self.attacker_num)]
		client_mask = [[] for _ in range(0,self.client_num + self.attacker_num)]

		class_every_task = int((public_class_num+self.private_class_num)/self.task_num)
		dirichlet_perclass = {}
		for i in class_public:
			a = np.random.dirichlet(np.ones(self.client_num), 1)
			while  (a < (1 / self.client_num / 2)).any():
				a = np.random.dirichlet(np.ones(self.client_num), 1)
			dirichlet_perclass[i] = a[0]

		for i in range(0,self.client_num):
			for j in range(0,self.task_num):
				index = []
				class_this_task = class_private[i][j*class_every_task: j*class_every_task+class_every_task]
				client_mask[i].append(class_this_task)
				for k in class_private[i][j*class_every_task:j*class_every_task+class_every_task]:
					if k in class_public:
						# 是公共类
						lenth = int(int(class_counts[k])*dirichlet_perclass[k][i])
						unused_indice = set(class_label[k])
						q = 0
						while q < lenth:
							random_index = random.choice(list(unused_indice))
							index.append(random_index)
							unused_indice.remove(random_index)
							q += 1
						class_label[k]=unused_indice
					else: #是私有类
						index.extend(class_label[k])
				random.shuffle(index)
				client_subset[i].append(CustomedSubset(trainset,index,trans,None))
				for attacker_index in range(self.attacker_num):
					if i == self.attacker_client_index[attacker_index]:
						client_subset[attacker_index + self.client_num].append(CustomedSubset(trainset,index,trans,None,attacker=True))
						client_mask[attacker_index + self.client_num].append(class_this_task)

		return client_subset,client_mask


	def process_testdata(self,surrogate_num):
		trans = build_transform(False,self.input_size)
		# self.Imagenet_R_test = Imagenet_R(root='C:/Users/Admin/datasets', train=False, download=True)
		testset = self.Imagenet_R_test
		# 100个类别的数据分给三个客户端使用

		class_counts = torch.zeros(200)  # 每个类的数量
		class_label = []  # 每个类的index
		for i in range(200):
			class_label.append([])
		j = 0
		for x, label in testset:
			class_counts[label] += 1
			class_label[label].append(j)
			j += 1
		# class_label 里保存了每个类的index

		surro_index =[]
		test_index = []
		for i in tqdm(range(200)):
			q = 0
			unused_indice = set(class_label[i])

			while q < surrogate_num:
				random_index = random.choice(list(unused_indice))
				surro_index.append(random_index)
				unused_indice.remove(random_index)
				q += 1
			test_index.extend(list(unused_indice))
		surrodata = CustomedSubset(testset,surro_index,trans,None)
		testdata = CustomedSubset(testset,test_index,trans,None)
		return surrodata,testdata

	def random_split_synchron(self):
		trans = build_transform(True,self.input_size)
		# self.Imagenet_R = Imagenet_R(root='C:/Users/Admin/datasets', train=True, download=True)
		trainset = self.Imagenet_R

		# 100个类别的数据分给三个客户端使用
		class_counts = torch.zeros(200) #每个类的数量
		class_label = [] # 每个类的index
		for i in range(200):
			class_label.append([])
		j = 0
		for x, label in tqdm(trainset):
			class_counts[label] += 1
			class_label[label].append(j)
			j += 1
		# class_label 里保存了每个类的index
		# 分类

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
					len = int(int(class_counts[k]) * random.uniform(0.3, 0.85))
					unused_indice = set(class_label[k])
					q = 0
					while q < len:
						random_index = random.choice(list(unused_indice))
						index.append(random_index)
						unused_indice.remove(random_index)
						q += 1

					random.shuffle(index)
				client_subset[i].append(CustomedSubset(trainset, index, trans,None))

		return client_subset,client_mask
