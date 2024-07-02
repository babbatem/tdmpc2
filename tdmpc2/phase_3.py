# Unfinished code to implement a 3rd phase for Adaptation TD-MPC to retrain TDMPC

import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):

	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)


	# Init agent
	#loaded_agent = TDMPC2(cfg) 
	#loaded_agent.load(cfg.checkpoint, "/home/walter/Desktop/encoder_checkpoint_3240.pt") # TODO: Change to configuration file

	trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
	encoder_location =  "/home/walter/Desktop/encoder_checkpoint_3240.pt"
	trainer = trainer_cls(
		cfg=cfg,
		env=make_env(cfg),
		agent=TDMPC2(cfg, cfg.checkpoint, encoder_location),
		buffer=Buffer(cfg),
		logger=Logger(cfg),
	)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
