# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym.spaces import MultiDiscrete
from gym_microrts import microrts_ai
from gym_microrts.envs.new_vec_env import MicroRTSGridModeVecEnv
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecVideoRecorder
from torch.distributions.categorical import Categorical

from new_ppo_gridnet import Agent, MicroRTSStatsRecorder


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicroRTSGridModeVecEnv",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=100000000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, the game will have partial observability')
    parser.add_argument('--n-minibatch', type=int, default=4,
        help='the number of mini batch')
    parser.add_argument('--num-bot-envs', type=int, default=0,
        help='the number of bot game environment; 16 bot envs measn 16 games')
    parser.add_argument('--num-selfplay-envs', type=int, default=0,
        help='the number of self play envs; 16 self play envs means 8 games')
    parser.add_argument('--num-steps', type=int, default=256,
        help='the number of steps per game environment')
    parser.add_argument("--agent-model-path", type=str, default="agent.pt",
        help="the path to the agent's model")
    parser.add_argument('--ai', type=str, default="lightRushAI",
        help='the number of steps per game environment')

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()

    # TRY NOT TO MODIFY: setup the environment
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.prod_mode:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=experiment_name,
            monitor_gym=True,
            save_code=True,
        )
        CHECKPOINT_FREQUENCY = 10

    # TRY NOT TO MODIFY: seeding
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    ais = []
    if args.ai:
        ais = [eval(f"microrts_ai.{args.ai}")]
        print(ais)
    envs = MicroRTSGridModeVecEnv(
        num_bot_envs=len(ais),
        num_selfplay_envs=args.num_selfplay_envs,
        partial_obs=args.partial_obs,
        max_steps=5000,
        render_theme=2,
        ai2s=ais,
        map_paths=["maps/16x16/basesWorkers16x16A.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 4.0, 4.0, 4.0, 0.2, 0.2, 1.0]),
    )
    envs = MicroRTSStatsRecorder(envs)
    envs = VecMonitor(envs)
    args.num_envs = len(ais) + args.num_selfplay_envs
    if args.capture_video:
        envs = VecVideoRecorder(
            envs, f"videos/{experiment_name}", record_video_trigger=lambda x: x % 100000 == 0, video_length=2000
        )
    assert isinstance(
        envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage for epoch data
    mapsize = 16 * 16
    action_space_shape = (mapsize, len(envs.action_plane_space.nvec))
    invalid_action_shape = (mapsize, envs.action_plane_space.nvec.sum())

    obs = torch.zeros((args.num_steps, args.num_envs) +
                      envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) +
                          action_space_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    invalid_action_masks = torch.zeros(
        (args.num_steps, args.num_envs) + invalid_action_shape).to(device)
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # Note how `next_obs` and `next_done` are used; their usage is equivalent to
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = 10000

    # CRASH AND RESUME LOGIC:
    starting_update = 1
    agent.load_state_dict(torch.load(args.agent_model_path))
    agent.eval()

    print("Model's state_dict:")
    for param_tensor in agent.state_dict():
        print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
    total_params = sum([param.nelement() for param in agent.parameters()])
    print("Model's total parameters:", total_params)

    for update in range(starting_update, num_updates + 1):
        # TRY NOT TO MODIFY: prepare the execution of the game.
        for step in range(0, args.num_steps):
            envs.render()
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                invalid_action_masks[step] = torch.tensor(
                    np.array(envs.get_action_mask())).to(device)
                action, logproba, _, _, vs = agent.get_action_and_value(
                    next_obs, envs=envs, invalid_action_masks=invalid_action_masks[
                        step], device=device
                )
                values[step] = vs.flatten()

            actions[step] = action
            logprobs[step] = logproba
            try:
                next_obs, rs, ds, infos = envs.step(
                    action.cpu().numpy().reshape(envs.num_envs, -1))
                next_obs = torch.Tensor(next_obs).to(device)
            except Exception as e:
                e.printStackTrace()
                raise
            rewards[step], next_done = torch.Tensor(
                rs).to(device), torch.Tensor(ds).to(device)

            for info in infos:
                if "episode" in info.keys():
                    # print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                    # writer.add_scalar("charts/episode_reward", info["episode"]["r"], global_step)
                    # for key in info["microrts_stats"]:
                    #     writer.add_scalar(f"charts/episode_reward/{key}", info["microrts_stats"][key], global_step)
                    # break

                    print("against", args.ai,
                          info["microrts_stats"]["WinLossRewardFunction"])

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        run.log({"charts/learning_rate": optimizer.param_groups[0]["lr"], "charts/update": update, "charts/sps": int(
            global_step / (time.time() - start_time))}, step=global_step)

    # envs.close()
    # writer.close()
