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

torch.backends.cudnn.benchmark = True


@hydra.main(config_name="config", config_path=".")
def train(cfg: dict):
    """
    Script for training the adaptation module for a single-task / multi-task TD-MPC2 checkpoint.

    Most relevant args:
            `task`: task name (or mt30/mt80 for multi-task evaluation)
            `model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
            `checkpoint`: path to model checkpoint to load
            `eval_episodes`: number of episodes to evaluate on per task (default: 10)
            `save_video`: whether to save a video of the evaluation (default: True)
            `seed`: random seed (default: 1)

    See config.yaml for a full list of args.

    Example usage:
    ````
            $ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
            $ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
            $ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
    ```
    """
    project_name = cfg.get("wandb_project", "none")
    entity_name = cfg.get("wandb_entity", "none")
    
    wandb.init(
            project=project_name,
            entity=entity_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            name="Adaptation test_longer_run",
            monitor_gym=True,
            save_code=True,
        )
    
    assert torch.cuda.is_available()
    assert cfg.adapt_episodes > 0, "Must create adaptation training for at least 1 episode."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored(f"Task: {cfg.task}", "blue", attrs=["bold"]))
    print(
        colored(
            f'Model size: {cfg.get("model_size", "default")}', "blue", attrs=["bold"]
        )
    )
    print(colored(f"Checkpoint: {cfg.checkpoint}", "blue", attrs=["bold"]))
    if not cfg.multitask and ("mt80" in cfg.checkpoint or "mt30" in cfg.checkpoint):
        print(
            colored(
                "Warning: single-task evaluation of multi-task models is not currently supported.",
                "red",
                attrs=["bold"],
            )
        )
        print(
            colored(
                "To evaluate a multi-task model, use task=mt80 or task=mt30.",
                "red",
                attrs=["bold"],
            )
        )

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
        # TODO: Change history to be defined in the config file
        history_length = 20
        # Create the encoder
        adapt_enc = create_encoder(cfg, obs_shape, history_length)

        optimizer = torch.optim.Adam(adapt_enc.parameters())
        for i in range(cfg.adapt_episodes):
            
            obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
            
            # Define history of obs
            # TODO: Delete obs in the array over time as more come in
            obs_ep = []
            obs_ep.append(obs)
            epoch_loss = 0.0
            batches = 0

            while not done:
                action = agent.act(obs, t0=t == 0, task=task_idx)
                obs, reward, done, info = env.step(action)
                #print(obs)
                ep_reward += reward
            
                # NOTE: new code added for adaptation module
                obs_ep.append(obs)
                if (len(obs_ep) > history_length):
                    optimizer.zero_grad()
                    obs_flat = torch.cat(obs_ep[-history_length:], dim =0)
                    new_z = adapt_enc(obs_flat).to(agent.device)
                    old_z = obtain_z (agent, obs, task_idx).to(agent.device)
                    loss = nn.MSELoss()(old_z, new_z)  # Calculate MSE loss
                    epoch_loss += loss.item()
                    batches += 1
                    loss.backward()  # Backpropagation
                    optimizer.step()  # Optimization step
                    #if t%1 == 0:
                         #print(f'Epoch [{i+1}/{cfg.adapt_episodes}], Loss: {loss.item():.6f}')
                t += 1
            print(f'Episode Length {t}')
            # Calculate average loss over the epoch
            avg_epoch_loss = epoch_loss / batches

            # Log average loss with wandb
            wandb.log({"loss": avg_epoch_loss})
            # Print average loss
            print(f'Epoch [{i + 1}/{cfg.adapt_episodes}], Average Loss: {avg_epoch_loss:.6f}')
        # Finish wandb run
        wandb.finish()


def obtain_z (agent, obs, task_idx):
    task_t = task_idx
    obs_t = obs.to(agent.device, non_blocking=True)
    if task_t is not None:
        task_t = torch.tensor([task_t], device=agent.device)
    z_0 = agent.model.encode(obs_t, task_t)
    return z_0.detach() # Teacher Z, detach so doesn't train

def create_encoder (cfg, obs_shape, history_length):
    #print( cfg.latent_dim)
    #print(cfg.enc_dim)
    # Changed encoder dim to be 1 hidden layers
    enc_mlp = layers.mlp(
                torch.prod(torch.tensor(obs_shape))*history_length + cfg.task_dim,
                max(2 - 1, 1) * [cfg.enc_dim],
                cfg.latent_dim,
                act=layers.SimNorm(cfg),
            )
    return enc_mlp


if __name__ == "__main__":
    train()
