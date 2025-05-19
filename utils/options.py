import argparse
import yaml

def create_args():
	
	# This function prepares the variables shared across demo.py
	parser = argparse.ArgumentParser()

	# Standard Args
	parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
						 help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
	parser.add_argument('--log_dir', type=str, default="outputs/out",
						 help="Save experiments results in dir for future plotting!")
	parser.add_argument('--learner_type', type=str, default='default', help="The type (filename) of learner")
	parser.add_argument('--learner_name', type=str, default='NormalNN', help="The class name of learner")
	parser.add_argument('--debug_mode', type=int, default=0, metavar='N',
						 help="activate learner specific settings for debug_mode")
	parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
	parser.add_argument('--overwrite', type=int, default=0, metavar='N', help='Train regardless of whether saved model exists')

	# CL Args		  
	parser.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
	parser.add_argument('--upper_bound_flag', default=False, action='store_true', help='Upper bound')
	parser.add_argument('--memory', type=int, default=0, help="size of memory for replay")
	parser.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
	parser.add_argument('--DW', default=False, action='store_true', help='dataset balancing')
	parser.add_argument('--prompt_param', nargs="+", type=int, default=[1, 1, 1],
						 help="e prompt pool size, e prompt length, g prompt length")
 
	# Arch params
	parser.add_argument('--mu', type=float, default=0.0)
	parser.add_argument('--beta', type=float, default=0.0)
	parser.add_argument('--eps', type=float, default=0.0)
	
 	# fed args
	parser.add_argument('--fedMoon', nargs="+", type=int, default=[2],
					 	 help="Fed ConMon 2 is the fedmoon version , 1 is the triplet loss + 1 ortho replacement, 3 is considering fedmoon with 1 ortho replacement")

	# Config Arg
	parser.add_argument('--config', type=str, default="configs/cifar-100_prompt.yaml",
						 help="yaml experiment config input")

	return parser
  
def get_args(argv):
	parser=create_args()
	args = parser.parse_args(argv)
	config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
	config.update(vars(args))
	return argparse.Namespace(**config)