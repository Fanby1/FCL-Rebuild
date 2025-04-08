from dataloaders import Cifar100_Spliter
from utils.utils import set_seed
from utils.utils import federated_average
from torch.utils.data import Subset
import sys

if __name__ == "__main__":

 
	set_seed(42)

	spliter = Cifar100_Spliter(client_num=5, task_num=5, private_class_num=15, input_size=224, path='C:/Users/Admin/datasets')
	client_subset,client_mask = spliter.random_split()

	print(client_mask)

	# del client_subset,client_mask,spliter

	from trainer import Trainer
	from utils.options import get_args
	args = get_args(sys.argv[1:])

	import os
	from utils.utils import Logger
	import sys
	import yaml
	import numpy as np
	# duplicate output stream to output file
	if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
	log_out = args.log_dir + '/output.log'
	sys.stdout = Logger(log_out)

	# save args
	with open(args.log_dir + '/args.yaml', 'w') as yaml_file:
		yaml.dump(vars(args), yaml_file, default_flow_style=False)

	metric_keys = ['acc','time']
	save_keys = ['global', 'client-1', 'client-2', 'client-3', 'client-4', 'client-5']
	global_only = ['time']
	avg_metrics = {}
	for mkey in metric_keys: 
		avg_metrics[mkey] = {}
		for skey in save_keys: avg_metrics[mkey][skey] = [None] * 10
	
	# load results
	if args.overwrite:
		start_r = 0
	else:
		try:
			for mkey in metric_keys: 
				for skey in save_keys:
					if (not (mkey in global_only)) or (skey == 'global'):
						save_file = args.log_dir+'/results-'+mkey+'/'+skey+'.yaml'
						if os.path.exists(save_file):
							with open(save_file, 'r') as yaml_file:
								yaml_result = yaml.safe_load(yaml_file)
								avg_metrics[mkey][skey] = np.asarray(yaml_result['history'])
			# next repeat needed
			start_r = avg_metrics[metric_keys[0]][save_keys[0]].shape[-1]
			# extend if more repeats left
			if start_r < args.repeat:
				max_task = avg_metrics['acc']['global'].shape[0]
				for mkey in metric_keys: 
					avg_metrics[mkey]['global'] = np.append(avg_metrics[mkey]['global'], np.zeros((max_task,args.repeat-start_r)), axis=-1)
					if (not (mkey in global_only)):
						avg_metrics[mkey]['pt'] = np.append(avg_metrics[mkey]['pt'], np.zeros((max_task,max_task,args.repeat-start_r)), axis=-1)
						avg_metrics[mkey]['pt-local'] = np.append(avg_metrics[mkey]['pt-local'], np.zeros((max_task,max_task,args.repeat-start_r)), axis=-1)
		except:
			start_r = 0
	
	from torch.utils.data import random_split

	client_data_train = []
	client_data_val = []
	for subset in client_subset:
		temp_train = []
		temp_val =[]
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
	seed = 42
 
	# culculate global data and mask
	global_datas = []
	global_masks = []
	for i in range(5):
		global_masks.append([])
		global_datas.append([])
		for j in range(5):
			global_masks[i].extend(client_mask[i][j])
		global_masks[i] = list(set(global_masks[i]))
  
		for idx in range(len(test_data.indices)):
			if test_data.targets[idx] in global_masks[i]:
				global_datas[i].append(idx)
		global_datas[i] = Subset(test_data, global_datas[i])
   
 
	global_trainer = Trainer(args, seed, metric_keys, save_keys, train_dataset=surro_data, test_dataset=test_data,
                        	validate_dataset=global_datas, client_index=0, class_mask=global_masks)
	for i in range(5):
		trainer = Trainer(args, seed, metric_keys, save_keys, train_dataset=client_data_train[i], test_dataset=test_data, 
			validate_dataset=client_data_val[i], client_index=1 + i, class_mask=client_mask[i])
		learner_pool.append(trainer)
	# for task in range(2):
	# 	for comunication_round in range(5):
    #  		# distribute model to clients     
	# 		for j in range(5):
	# 			learner_pool[j].learner.model.load_state_dict(global_trainer.learner.model.state_dict())
	# 			learner_pool[j].curr_task = task
	# 			learner_pool[j].conmunication_round = comunication_round
    
	# 		# train model
	# 		for j in range(5):
	# 			learner_pool[j].train(avg_metrics)
    
	# 		# aggregate model
	# 		average_weight = federated_average(learner_pool, [1, 1, 1, 1, 1])
	# 		global_trainer.learner.model.load_state_dict(average_weight)
	# 		model_save_dir = global_trainer.model_top_dir + f'/models/repeat-{seed + 1}/client-{0}/task-{task+1}//comunication_round-{comunication_round}/'
	# 		if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
	# 		global_trainer.learner.save_model(model_save_dir)
	# 		global_trainer.learner.load_model(model_save_dir)

   
	global_trainer.evaluate(avg_metrics)
	# trainer.evaluate(avg_metrics)