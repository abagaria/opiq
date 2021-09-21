import time
import torch
import numpy as np
import matplotlib.pyplot as plt


def chunked_inference(func, data, to_tensor=True, device=torch.device("cuda"), chunk_size=1000):
    """ Apply and aggregate func on chunked versions of data. """

    num_chunks = int(np.ceil(data.shape[0] / chunk_size))

    if num_chunks == 0:
        return 0.

    current_idx = 0
    chunks = np.array_split(data, num_chunks, axis=0)
    values = np.zeros((data.shape[0],))

    for data_chunk in chunks:
        
        if to_tensor:
            data_chunk = torch.from_numpy(data_chunk).float().to(device)

        value_chunk = func(data_chunk)
        current_chunk_size = len(data_chunk)
        values[current_idx:current_idx + current_chunk_size] = value_chunk
        current_idx += current_chunk_size

    return values


def visualize_value_function(agent, episode, experiment_name, seed):
    t0 = time.time()

    num_samples = min(agent.replay_buffer.num_in_buffer, 5000)
    batch = agent.replay_buffer.sample(num_samples)
    obs = batch[0]
    pos = batch[-1]["pos"]

    values = chunked_inference(agent.value_function, obs)

    fname = "{}_{}/value_function_episode_{}.png".format(experiment_name, seed, episode)
    plot_against_position(pos, values, episode, fname)

    print("Took {}s to plot intrinsic value function".format(time.time() - t0))


def visualize_intrinsic_reward_function(agent, episode, experiment_name, seed):
    t0 = time.time()

    frames = agent.replay_buffer.obs[:agent.replay_buffer.next_idx][:, -1]
    positions = agent.replay_buffer.pos[:agent.replay_buffer.next_idx]
    intrinsic_rewards = chunked_inference(agent.intrinsic_reward_function, frames, to_tensor=False)

    fname = "{}_{}/intrinsic_reward_episode_{}.png".format(episode, experiment_name, seed)
    plot_against_position(positions, intrinsic_rewards, episode, fname)

    print("Took {}s to plot intrinsic reward function".format(time.time() - t0))


def plot_against_position(positions, values, episode, fname):
    x, y, r = [], [], []

    for pos, r_int in zip(positions, values):
        if pos[0] is not None and pos[1] is not None:
            x.append(pos[0])
            y.append(pos[1])
            r.append(r_int)

    plt.scatter(x, y, c=r)
    plt.title("Episode {}".format(episode))
    plt.colorbar()
    plt.savefig("plots/{}".format(fname))
    plt.close()
