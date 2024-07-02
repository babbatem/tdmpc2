# Code written to test whether priviliged information could be determined from observation and action sequence


import os
import wandb

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


os.environ["MUJOCO_GL"] = "egl"
import warnings

warnings.filterwarnings("ignore")

import hydra
import imageio
import numpy as np
import torch
import torch.nn as nn
from termcolor import colored
from omegaconf import OmegaConf

# Added Layers import
from common import layers

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

torch.backends.cudnn.benchmark = True


@hydra.main(config_name="config", config_path=".")
def train(cfg: dict):
 
    
    assert torch.cuda.is_available()
    assert cfg.adapt_episodes > 0, "Must create adaptation training for at least 1 episode."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    # Make environment
    env = make_env(cfg)

    # Load agent
    agent = TDMPC2(cfg)
    assert os.path.exists(
        cfg.checkpoint
    ), f"Checkpoint {cfg.checkpoint} not found! Must be a valid filepath."
    agent.load(cfg.checkpoint)

    tasks = cfg.tasks if cfg.multitask else [cfg.task]
    for task_idx, task in enumerate(tasks):
        if not cfg.multitask:
            task_idx = None

        # TODO: Replace with configuration variable
        obs_shape = 21
        action_shape = 3
        # TODO: Create config variable for the priviliged size
        priv_size = 4
        # TODO: Change history to be defined in the config file
        history_length = 40
        

        obs_flat_array = []
        priv_array = []
        for i in range(200):
            
            obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
            
            # Define history of obs
            # TODO: Delete obs in the array over time as more come in
            obs_ep = []
            obs_list = obs.tolist()
            obs_reg = obs_list[0:-priv_size] + [0,0,0]
            
            
            obs_ep.append(obs_reg)
            
            
            while not done:
                action = agent.act(obs, t0=t == 0, task=task_idx)
                obs, reward, done, info = env.step(action)
                
                #print(obs)
                ep_reward += reward
                
                obs_list = obs.tolist()
                # NOTE: new code added for adaptation module
                # Remove the priviliged information
                obs_reg = obs_list[0:-priv_size] + action.tolist()
                
                obs_ep.append(obs_reg)
                obs_flat = []
                
                if (len(obs_ep) > history_length):
                    obs_flat = np.array(obs_ep[-history_length:]).flatten().tolist()
                    
                else:
                    padding = [[0]*len(obs_ep[0])] * (history_length - len(obs_ep))
                    obs_flat = np.array(padding + obs_ep).flatten().tolist()
                obs_flat_array.append(obs_flat)
                priv_array.append(obs[-priv_size:].tolist())
            print(i)
    
    X_train, X_test, y_train, y_test = train_test_split(obs_flat_array, priv_array, test_size=0.2, random_state=22)

    # Check if there are any NaN or infinite values in the data
    print("NaN in X_train:", np.isnan(X_train).any())
    print("Inf in X_train:", np.isinf(X_train).any())
    print("NaN in y_train:", np.isnan(y_train).any())
    print("Inf in y_train:", np.isinf(y_train).any())

    # Create the model
    nn_model = MLPRegressor(hidden_layer_sizes=(64,), activation='relu', solver='adam', max_iter=500)
    # Train the model
    nn_model.fit(X_train, y_train) 

    # Predict and evaluate
    y_pred = nn_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    
    
    '''
    # Creating and training the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    


    # Predicting on the test set
    y_pred = model.predict(X_test)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    '''        

if __name__ == "__main__":
    train()
