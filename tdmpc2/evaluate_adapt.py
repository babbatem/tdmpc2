# NOTE: This script is based on the evaluate.py script provided by TDMPC


import os
import gc

os.environ["MUJOCO_GL"] = "egl"
import warnings

warnings.filterwarnings("ignore")

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True

torch.cuda.memory_summary(device=None, abbreviated=False)
@hydra.main(config_name="config", config_path=".")
def evaluate(cfg: dict):
    """
    Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

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
    assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0, "Must evaluate at least 1 episode."
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

    import wandb
    from omegaconf import OmegaConf    
    # project_name = cfg.get("wandb_project", "none")
    # entity_name = cfg.get("wandb_entity", "none")
    
    wandb.init(
            project="adaptation",
            entity="robin-abba",
            config=OmegaConf.to_container(cfg, resolve=True),
            name="Adaptation_module_eval",
            monitor_gym=True,
            save_code=True,
        )

    # Make environment
    env = make_env(cfg)

    # Load agent
    agent = TDMPC2(cfg)
    assert os.path.exists(
        cfg.checkpoint
    ), f"Checkpoint {cfg.checkpoint} not found! Must be a valid filepath."
   
    # Loads previous model plus the new encoder
    agent.load(cfg.checkpoint, "/home/dk/encoder_checkpoint_10000.pt")
    #agent.load(cfg.checkpoint)
    # TODO: Change to configuration file variable
    priv_size = 4
    # TODO: Change history to be defined in the config file
    history_length = 20

    # Evaluate
    if cfg.multitask:
        print(
            colored(
                f"Evaluating agent on {len(cfg.tasks)} tasks:", "yellow", attrs=["bold"]
            )
        )
    else:
        print(colored(f"Evaluating agent on {cfg.task}:", "yellow", attrs=["bold"]))
    if cfg.save_video:
        video_dir = os.path.join(cfg.work_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
    scores = []
    tasks = cfg.tasks if cfg.multitask else [cfg.task]
    with torch.no_grad():
        for task_idx, task in enumerate(tasks):
            if not cfg.multitask:
                task_idx = None
            ep_rewards, ep_successes = [], []
            for i in range(cfg.eval_episodes): # TODO: Add a parameter in the config for the number of epochs
                obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0

                obs_reg = torch.cat((torch.tensor(obs[0:-priv_size]), torch.tensor([0,0,0])), dim=0)
                obs_ep = []
                obs_ep.append(obs_reg)



                if cfg.save_video:
                    frames = [env.render()]
                while not done:
                    obs_flat = []
                    if(len(obs_ep) < history_length):
                        obs_flat = torch.cat([torch.zeros_like(obs_ep[0])] * (history_length - len(obs_ep)) + obs_ep, dim =0)   
                    else:
                        obs_flat = torch.cat(obs_ep[-history_length:], dim =0)
                    
                    #obs_flat = obs
                    
                    action = agent.act(obs_flat, t0=t == 0, task=task_idx)
                    
                    
                    
                    obs, reward, done, info = env.step(action)
                    ep_reward += reward
                    if done:
                        wandb.log({"Episode Success": info["success"], "Episode Reward": ep_reward})

                    obs_reg = torch.cat((torch.tensor(obs[0:-priv_size]), torch.tensor(action)), dim=0)
                    obs_ep.append(obs_reg)

                    if cfg.save_video:
                        update_markers(agent, env, cfg, obs_flat, t == 0, task_idx)
                        frames.append(env.render())

                    t += 1


                obs_ep = []
                torch.cuda.empty_cache()    
                gc.collect()
                ep_rewards.append(ep_reward)
                ep_successes.append(info["success"])
                print(info["success"])
                avg_success = np.mean(ep_successes)
                avg_reward = np.mean(ep_rewards)
            
                wandb.log({
                    f"Average Success Rate {task}": avg_success,
                    f"Average Reward {task}": avg_reward
                })
                if cfg.save_video:
                    imageio.mimsave(
                        os.path.join(video_dir, f"{task}-{i}.mp4"), frames, fps=15
                    )
            
            # ep_rewards = np.mean(ep_rewards)
            # ep_successes = np.mean(ep_successes)

            if cfg.multitask:
                scores.append(
                    ep_successes * 100 if task.startswith("mw-") else ep_rewards / 10
                )
            print(
                colored(
                    f"  {task:<22}" f"\tR: {avg_reward:.01f}  " f"\tS: {avg_success:.02f}",
                    "yellow",
                )
            )
        if cfg.multitask:
            print(
                colored(
                    f"Normalized score: {np.mean(scores):.02f}", "yellow", attrs=["bold"]
                )
            )


def update_markers(agent, env, cfg, obs, t0, task_idx):
    task_t = task_idx
    obs_t = obs.to(agent.device, non_blocking=True)
    if task_t is not None:
        task_t = torch.tensor([task_t], device=agent.device)

    z_0 = agent.model.encode(obs_t, task_t)
    z_s = torch.empty(
        cfg.num_pi_trajs + 1,
        cfg.horizon + 1,
        cfg.latent_dim,
        device=agent.device,
    )

    # sample TDMPC planning
    plan = agent.plan(z_0, t0=t0, eval_mode=True, task=task_idx)
    z_s[0, 0] = z_0
    for i, action in enumerate(plan):
        z_s[0, i + 1] = agent.model.next(z_s[0, i], action, task_idx)

    # sample all policy trajectories
    z_s[1:] = agent.sample_trajectories(z_0, task=task_idx)

    env.unwrapped.update_trajectories(agent.model.decode(z_s).detach().cpu())


if __name__ == "__main__":
    evaluate()
