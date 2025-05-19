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

	seed = 1999
	set_seed(seed)
	client_num = 5
	attacker_num = 0
 
	client_count = client_num + attacker_num
	client_weight = [1] * client_count
 
	comunication_round_count = 5
 
	task_count = 5
	# duplicate output stream to output file
	if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
	log_out = args.log_dir + '/output.log'
	sys.stdout = Logger(log_out)

	# save args
	with open(args.log_dir + '/args.yaml', 'w') as yaml_file:
		yaml.dump(vars(args), yaml_file, default_flow_style=False)

	metric_keys = ['task-1-acc', 'last-task-acc','time']
	save_keys = ['global']
 
	fed_trainer = FedTrainer(args, seed, metric_keys, save_keys, client_num, attacker_num, 
						  task_count, 40, client_count, comunication_round_count)
 
	fed_trainer.train()
 
	fed_trainer.evaluate()