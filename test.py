from utils.utils import set_seed
import sys
from trainer import Trainer
from utils.options import get_args
import os
from utils.utils import Logger
import sys
import yaml
from fed_trainer import FedTrainer

if __name__ == "__main__":
	args = get_args(sys.argv[1:])

	schedule = args.schedule
	seed = 2024
	set_seed(seed)
	client_num = 5
	attacker_num = 0 
	task_count = 5
 
	client_count = client_num + attacker_num
	client_weight = [1] * client_count
 
	comunication_round_count = schedule[1]

	# duplicate output stream to output file
	if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
	log_out = args.log_dir + '/output.log'
	sys.stdout = Logger(log_out)

	# save args
	with open(args.log_dir + '/args.yaml', 'w') as yaml_file:
		yaml.dump(vars(args), yaml_file, default_flow_style=False)

	metric_keys = ['task-1-acc', 'last-task-acc','time']
	save_keys = []
 
	fed_trainer = FedTrainer(args, seed, metric_keys, save_keys, client_num, attacker_num, 
						  task_count, 40, client_count, comunication_round_count, synchronize=False, path='C:/Users/Admin/datasets')
 
	fed_trainer.train()
 
	# fed_trainer.evaluate()