from dataloaders import Cifar100_Spliter, ImageNetR_Spliter
from utils.utils import set_seed
from utils.utils import federated_average
from torch.utils.data import Subset
import sys, copy
from trainer import Trainer
from utils.options import get_args
import os
from utils.utils import Logger
import sys
import yaml
import numpy as np
import torch

if __name__ == "__main__":
	args = get_args(sys.argv[1:])

	seed = 42
	set_seed(seed)
	client_num = 5
	attacker_num = 5
 
	client_count = client_num + attacker_num
	client_weight = [1] * client_count
 
	comunication_round_count = 5
 
	task_count = 5

	spliter = Cifar100_Spliter(client_num=client_num, attacker_num=attacker_num, 
                             task_num=task_count, private_class_num=15, input_size=224, path='C:/Users/Admin/datasets')
	client_subset,client_mask = spliter.random_split()

	print(client_mask)

	# duplicate output stream to output file
	if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
	log_out = args.log_dir + '/output.log'
	sys.stdout = Logger(log_out)

	# save args
	with open(args.log_dir + '/args.yaml', 'w') as yaml_file:
		yaml.dump(vars(args), yaml_file, default_flow_style=False)

	metric_keys = ['task-1-acc', 'last-task-acc','time']
	save_keys = ['global']
 
	for i in range(client_count):
		save_keys.append(f'client-{i + 1}')

	avg_metrics = {}
	for mkey in metric_keys: 
		avg_metrics[mkey] = {}
		for skey in save_keys: 
			avg_metrics[mkey][skey] = [None] * 5
	
	from torch.utils.data import random_split

	client_data_train = []
	client_data_val = []
	for subset in client_subset:
		temp_train = []
		temp_val = []
		for data in subset:
			# data, _ = random_split(data, [int(len(data) * 0.1), len(data) - int(len(data) * 0.1)])
			train_dataset, val_dataset = random_split(data, [int(len(data) * 0.7), len(data) - int(len(data) * 0.7)])
			temp_train.append(train_dataset)
			temp_val.append(val_dataset)
		client_data_train.append(temp_train)
		client_data_val.append(temp_val)
  
	# subset, client_mask = spliter.random_split_synchron()
	
	surro_data, test_data = spliter.process_testdata(5)
	learner_pool = []
 
	try:
		spliter.Imagenet_R = None
		spliter.Imagenet_R_test = None
	except:
		pass


 
	# culculate global data and mask
	global_datas = []
	global_masks = []
	for i in range(task_count):
		global_masks.append([])
		global_datas.append([])
		for j in range(client_count):
			global_masks[i].extend(client_mask[j][i])
		global_masks[i] = list(set(global_masks[i]))
  
		for idx in range(len(test_data.indices)):
			if test_data.targets[idx] in global_masks[i]:
				global_datas[i].append(idx)
		global_datas[i] = Subset(test_data, global_datas[i])
   

				
	global_trainer = Trainer(args, seed, metric_keys, save_keys, train_dataset=surro_data, test_dataset=test_data,
							validate_dataset=global_datas, client_index=0, class_mask=global_masks)
	

	for i in range(client_count):
		trainer = Trainer(args, seed, metric_keys, save_keys, train_dataset=client_data_train[i], test_dataset=test_data, 
			validate_dataset=client_data_val[i], client_index=1 + i, class_mask=client_mask[i])
		learner_pool.append(trainer)
  
	previous_global_trainer = None
 
	
 
	# time.sleep(1000)
  
	
	for task in range(task_count):
		# increment task id in prompting modules
		previous_global_trainer = copy.deepcopy(global_trainer.learner)
		global_trainer.before_task(None, None, task, comunication_round_count, 0)
		try:
			if global_trainer.learner.model.module.prompt is not None:
				global_trainer.learner.model.module.prompt.process_task_count()
		except:
			if global_trainer.learner.model.prompt is not None:
				global_trainer.learner.model.prompt.process_task_count()
	
		for comunication_round in range(comunication_round_count):
			for j in range(client_count):
				# torch.cuda.memory._record_memory_history()               # 开始记录
				# # distribute model to clients	 
				if comunication_round == 0 and task != 0:
					learner_pool[j].before_task(global_trainer.learner.model, previous_global_trainer.model, task, comunication_round, j)
				else:
					learner_pool[j].before_task(global_trainer.learner.model, None, task, comunication_round, j)

				# train model
				learner_pool[j].train(avg_metrics)
				# torch.cuda.memory._dump_snapshot("my_snapshot.pickle")   # 保存文件
				# torch.cuda.memory._record_memory_history(enabled=None)   # 终止记录
				# exit(0)
				
	
			# aggregate model
			average_weight = federated_average(learner_pool, client_weight)
			global_trainer.learner.model.load_state_dict(average_weight)
			model_save_dir = global_trainer.model_top_dir + f'/models/repeat-{seed + 1}/client-{0}/task-{task+1}/comunication_round-{comunication_round}/'
			if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
			global_trainer.learner.save_model(model_save_dir)
 
			
   
	for client_index in range(client_count):
		learner_pool[client_index].evaluate(avg_metrics, comunication_round=comunication_round_count - 1)
	print(f"-------------------------global--------------------------")
	global_trainer.evaluate(avg_metrics, comunication_round=comunication_round_count - 1)
	
	print(avg_metrics)