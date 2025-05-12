import numpy as np
import random

from typing import TypeVar, Sequence
from torch.utils.data import Dataset
from PIL import Image

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

class CustomedSubset(Dataset[T_co]):
	r"""
	Subset of a dataset at specified indices.

	Args:
		dataset (Dataset): The whole Dataset
		indices (sequence): Indices in the whole set selected for subset
	"""
	dataset: Dataset[T_co]
	indices: Sequence[int]

	def __init__(self, dataset: Dataset[T_co], indices: Sequence[int], trans, show_sample = None, attacker:bool = False) -> None:

		self.indices = indices
		self.data = []
		self.targets = []
		self.transform_pretrain = trans
		self.target_transform = None

		self.show_sample = show_sample

		for i in self.indices:
			try:
				self.data.append(dataset.path_list[i])
			except:
				self.data.append(dataset.data[i])
			self.targets.append(dataset.targets[i])
		self.data = np.array(self.data)
		self.targets = np.array(self.targets)

		self.attacker = attacker

	def __getitem__(self, idx):
		img, target = self.data[idx], self.targets[idx]
		if self.attacker:
			img = self.data[random.choice(range(len(self.data)))]
		if isinstance(img, str):
			img = np.array(Image.open(img).convert('RGB'))
		img = Image.fromarray(img)

		if self.transform_pretrain is not None:
			img = self.transform_pretrain(img)

		if self.target_transform is not None:
			target = self.target_transform(target)
		return img,target

	def get_test_sample(self):
		img = self.show_sample

		if self.transform_pretrain is not None:
			img_pre = self.transform_pretrain(img)

		return img


	def __len__(self):
		return len(self.indices)