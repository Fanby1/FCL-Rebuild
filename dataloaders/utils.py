import torch
import numpy as np
import os
from PIL import Image
from torchvision.transforms import transforms

def accuracy(output, target, topk=(1,)):
	### 返回的是正确的个数
	maxk = min(max(topk), output.size()[1])
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.reshape(1, -1).expand_as(pred))

	return [correct[:min(k, maxk)].reshape(-1).float().sum(0)  for k in topk],batch_size


def global_distillation_loss(output,outputs):
	mse = torch.nn.MSELoss(reduction='sum')
	total_loss = 0
	for i in outputs:
		loss=mse(output,i)
		total_loss +=loss
	return total_loss

def build_transform(is_train,input_size) -> transforms.Compose: 
	resize_im = input_size > 32
	if is_train:
		scale = (0.05, 1.0)
		ratio = (3. / 4., 4. / 3.)
		transform = transforms.Compose([
			transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.ToTensor(),
		])
		return transform

	t = []
	if resize_im:
		size = int((256 / 224) * input_size)
		t.append(
			transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
		)
		t.append(transforms.CenterCrop(input_size))
	t.append(transforms.ToTensor())
	return transforms.Compose(t)

def process_file(fpath, folder, idx):
	"""加载并转换单张图片，同时返回对应的标签"""
	folder_path = os.path.join(fpath, folder)
	X = []
	y = []
	path = []
	for ims in os.listdir(folder_path):
		img_path = os.path.join(folder_path, ims)
		try:
			# img = np.array(Image.open(img_path).convert('RGB'))
			X.append(None)
			y.append(idx)
			path.append(img_path)
		except Exception as e:
			print(f"Error processing {img_path}: {e}")
	return X, y, path