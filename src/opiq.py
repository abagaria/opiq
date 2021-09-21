import pdb
import gym
import yaml
import time
import torch
import utils
import pickle
import random
import logging
import argparse
import numpy as np
import utils.plotting

from collections import deque
from agent.atari_dqn import DQN
from buffer.buffer import ReplayBuffer
from envs.env_wrapper import EnvWrapper
from trainer.dqn_train import DQNTrainer
from count.atari_count import AtariCount
from action.optimistic_action import OptimisticAction
from utils.dict2namedtuple import convert
from utils.logging import get_stats, configure_stats_logging, create_log_dir


class OPIQAgent:
    def __init__(self, obs_dtype, obs_scaling, config, seed, device):
        self.seed = seed
        self.config = config
        self.device = device 

        configure_stats_logging(
            str(seed) + "_" + config.name,
            log_interval=config.log_interval,
            sacred_info={},
            use_tb=config.tb,
        )
        
        self.stats = get_stats()
        self.logger = logging.getLogger("Main")

        self.agent = DQN(config)
        self.target_agent = DQN(config)
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.agent.to(device)
        self.target_agent.to(device)

        self.count_model = AtariCount(config)
        self.action_selector = OptimisticAction(self.count_model, config)

        self.replay_buffer = ReplayBuffer(
            size=config.buffer_size, frame_history_len=config.past_frames_input,
            obs_dtype=obs_dtype, obs_scaling=obs_scaling, args=config
        )

        self.trainer = DQNTrainer(agent=self.agent,
                                  target_agent=self.target_agent,
                                  args=config,
                                  count_model=self.count_model,
                                  buffer=self.replay_buffer,
                                  device=device)
        
        self.T = 0 
    
    def rollout(self, env, state, max_reward_so_far):
        assert isinstance(env, gym.Env), env
        assert isinstance(state, np.ndarray), state

        terminated = False
        episode_reward = 0.
        episode_length = 0
        intrinsic_episode_reward = 0.
        
        while not terminated: 

            # Store observation (a single frame) into replay buffer
            buffer_idx = self.replay_buffer.store_frame(state, pos=env.wrapped_env.get_player_xy())
            stacked_states = self.replay_buffer.encode_recent_observation()
            
            # Action selection
            with torch.no_grad():
                tensor_state = torch.tensor(stacked_states, device=self.device).unsqueeze(0)
                agent_output = self.agent(tensor_state)

            action, action_info = self.action_selector.select_actions(agent_output, 
                                                                      self.T,
                                                                      info={
                                                                        "state": tensor_state
                                                                    })
            
            # Execute agent action
            next_state, reward, terminated, info = env.step(action)   
            
            # Logging
            self.T += 1
            episode_length += 1
            episode_reward += reward

            # Manage the done flag
            terminal_to_store = terminated
            if "Steps_Termination" in info and info["Steps_Termination"]:
                self.logger.warning("Terminating because of episode limit")
                terminal_to_store = False

            # Count-based exploration bonus
            pseudo_count = self.count_model.visit(tensor_state, action)
            count_bonus = self.config.count_beta / np.sqrt(pseudo_count)
            intrinsic_reward = count_bonus

            intrinsic_episode_reward += intrinsic_reward
            
            # Add what happened to the buffer
            self.replay_buffer.store_effect(buffer_idx,
                                            action,
                                            reward - self.config.reward_baseline,
                                            intrinsic_reward,
                                            terminal_to_store,
                                            pseudo_count)

            # Update state
            state = next_state

            if terminated:
                if "Steps_Termination" in info and info["Steps_Termination"]:
                    buffer_idx = self.replay_buffer.store_frame(state, env.wrapped_env.get_player_xy())
                    self.replay_buffer.store_effect(buffer_idx, 0, 0, 0, True, 0, dont_sample=True)

                max_reward_so_far = max(max_reward_so_far, episode_reward)

                self.logger.warning("T: {:,}, Episode Reward: {:.2f}, Max Reward {:.2f}".format(self.T,
                                                                                                episode_reward,
                                                                                                max_reward_so_far))
            
            # Train if possible
            for _ in range(self.config.training_iters):
                sampled_batch = None

                if self.T % config.update_freq != 0:
                    # Only train every update_freq timesteps
                    continue
                if self.replay_buffer.can_sample(config.batch_size):
                    sampled_batch = self.replay_buffer.sample(config.batch_size, nstep=config.n_step)

                if sampled_batch is not None:
                    self.trainer.train(sampled_batch)
            
            # Update target networks if necessary
            if self.T % self.config.target_update_interval == 0:
                self.trainer.update_target_agent()

        return episode_reward, episode_length, max_reward_so_far

    def intrinsic_reward_function(self, states):
        """ Given a batch of states, return the corresponding intrinsic rewards. """
        
        def pre_process(x):
            """ Convert to float and divide each pixel by 255. """
            return x.astype(self.replay_buffer.obs_dtype) * self.replay_buffer.obs_scaling

        def counts2bonus(c):
            """ Action selection count based exploration bonus. """
            return self.config.optim_action_tau / (c + 1.0) ** self.config.optim_m

        assert isinstance(states, np.ndarray)
        states = pre_process(states)
        states_tensor = torch.as_tensor(states).float().to(self.device)
        action_counts = self.count_model.get_all_action_counts(states_tensor).transpose(1, 0)
        state_counts = action_counts.max(axis=1)
        assert state_counts.shape == (states.shape[0],), state_counts.shape
        return counts2bonus(state_counts)

    def value_function(self, states):
        def pre_process(x):
            x = torch.as_tensor(x).float() * self.replay_buffer.obs_scaling
            return x.to(self.device)

        with torch.no_grad():
            q_values = self.agent(pre_process(states))

        values = q_values.max(dim=1).values
        assert values.shape == (states.shape[0],), values.shape

        return values.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--experiment_name", type=str)
    args = parser.parse_args()
    
    create_log_dir("plots")
    create_log_dir("logs")
    create_log_dir("plots/{}_{}".format(args.experiment_name, args.seed))
    create_log_dir("logs/{}_{}".format(args.experiment_name, args.seed))
    _log_file = "logs/{}_{}/opiq_log.pkl".format(args.experiment_name, args.seed)

    config = yaml.load(open("src/config/montezuma.yaml", "r"))
    config = convert(config)

    environment = gym.make(config.env)
    num_actions = environment.action_space.n
    config = config._replace(num_actions=num_actions)
    state_shape = environment.observation_space.shape
    config = config._replace(state_shape=state_shape)
    environment = EnvWrapper(environment, debug=True, args=config)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    environment.seed(args.seed)

    opiq_agent = OPIQAgent(obs_dtype=getattr(environment.wrapped_env, "obs_dtype", np.float32),
                           obs_scaling=getattr(environment.wrapped_env, "obs_scaling", 1),
                           config=config,
                           seed=args.seed,
                           device=torch.device(args.device))

    t0 = time.time()
    current_step_number = 0
    max_episodic_reward = 0
    current_episode_number = 0
    num_training_steps = 13000000

    _log_steps = []
    _log_rewards = []
    _log_max_rewards = []

    while current_step_number < num_training_steps:
        s0 = environment.reset()
        episodic_reward, episodic_duration, max_episodic_reward = opiq_agent.rollout(environment, s0, max_episodic_reward)

        current_episode_number += 1
        current_step_number += episodic_duration

        _log_steps.append(current_step_number)
        _log_rewards.append(episodic_reward)
        _log_max_rewards.append(max_episodic_reward)

        with open(_log_file, "wb+") as f:
            episode_metrics = {
                            "step": _log_steps, 
                            "reward": _log_rewards,
                            "max_reward": _log_max_rewards
            }
            pickle.dump(episode_metrics, f)

        if current_episode_number > 0 and current_episode_number % 100 == 0:
            utils.plotting.visualize_value_function(opiq_agent, current_episode_number, args.experiment_name, args.seed)

    print("Finished after {} hrs".format((time.time() - t0) / 3600.))
