from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
import torch
import numpy as np
import yaml
import random
from trainer import Trainer
from utils.utils import Logger
from utils.options import get_args

if __name__ == '__main__':
	args = get_args(sys.argv[1:])

	# determinstic backend
	torch.backends.cudnn.deterministic=True

	# duplicate output stream to output file
	if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
	log_out = args.log_dir + '/output.log'
	sys.stdout = Logger(log_out)

	# save args
	with open(args.log_dir + '/args.yaml', 'w') as yaml_file:
		yaml.dump(vars(args), yaml_file, default_flow_style=False)
	
	metric_keys = ['acc','time',]
	save_keys = ['global', 'pt', 'pt-local']
	global_only = ['time']
	avg_metrics = {}
	for mkey in metric_keys: 
		avg_metrics[mkey] = {}
		for skey in save_keys: avg_metrics[mkey][skey] = []
	start_r = 0
	for r in range(start_r, args.repeat):

		print('************************************')
		print('* STARTING TRIAL ' + str(r+1))
		print('************************************')

		# set random seeds
		seed = r
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)

		# set up a trainer
		trainer = Trainer(args, seed, metric_keys, save_keys)

		# init total run metrics storage
		max_task = trainer.max_task
		if r == 0: 
			for mkey in metric_keys: 
				avg_metrics[mkey]['global'] = np.zeros((max_task,args.repeat))
				if (not (mkey in global_only)):
					avg_metrics[mkey]['pt'] = np.zeros((max_task,max_task,args.repeat))
					avg_metrics[mkey]['pt-local'] = np.zeros((max_task,max_task,args.repeat))

		# train model
		avg_metrics = trainer.train(avg_metrics)  

		# evaluate model
		avg_metrics = trainer.evaluate(avg_metrics)	

		# save results
		for mkey in metric_keys: 
			m_dir = args.log_dir+'/results-'+mkey+'/'
			if not os.path.exists(m_dir): os.makedirs(m_dir)
			for skey in save_keys:
				if (not (mkey in global_only)) or (skey == 'global'):
					save_file = m_dir+skey+'.yaml'
					result=avg_metrics[mkey][skey]
					yaml_results = {}
					if len(result.shape) > 2:
						yaml_results['mean'] = result[:,:,:r+1].mean(axis=2).tolist()
						if r>1: yaml_results['std'] = result[:,:,:r+1].std(axis=2).tolist()
						yaml_results['history'] = result[:,:,:r+1].tolist()
					else:
						yaml_results['mean'] = result[:,:r+1].mean(axis=1).tolist()
						if r>1: yaml_results['std'] = result[:,:r+1].std(axis=1).tolist()
						yaml_results['history'] = result[:,:r+1].tolist()
					with open(save_file, 'w') as yaml_file:
						yaml.dump(yaml_results, yaml_file, default_flow_style=False)

		# Print the summary so far
		print('===Summary of experiment repeats:',r+1,'/',args.repeat,'===')
		for mkey in metric_keys: 
			print(mkey, ' | mean:', avg_metrics[mkey]['global'][-1,:r+1].mean(), 'std:', avg_metrics[mkey]['global'][-1,:r+1].std())
	