import gym
from agent import DQNAgent
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from atari_utils import warp_deepmind, EpisodicLifeEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

episode = 400
env = gym.make('ALE/Pong-v5', render_mode='human')
env = warp_deepmind(env, episode_life=True, fire_reset=True, frame_stack=True, scale=False)
# env = EpisodicLifeEnv(env)
shape = (env.observation_space.shape[2], env.observation_space.shape[0], env.observation_space.shape[1])
agent = DQNAgent(shape, env.action_space.n, dueling=True, noisy=False, use_conv=True)
agent.eval_net.to(device)
agent.target_net.to(device)
agent.load('./models/Pong-v5_400.pth')
total_rewards = []
pgr_bar = tqdm(range(episode))
epsilon = 1

for eps in pgr_bar:
    state, info = env.reset()
    done = False
    total_reward = 0
    epsilon = max(0.1, epsilon - 2 / episode)
    while not done:
        action = agent.sample(state, 0.1)
        # next_state, reward, done, _, _ = env.step(action)
        next_state, reward, done, _, _ = env.step(action)
        if done:
            reward = -100
        total_reward += reward
        agent.remember(state, action, reward, next_state)
        if len(agent.replay_buffer) >= agent.replay_buffer_size:
            agent.learn()
            if eps % 10 == 0 or eps == episode and eps != 0:
                agent.save(f"models/Pong-v5_{eps}.pth")
        state = next_state
    total_rewards.append(total_reward)
    pgr_bar.set_description(f"Tota reward: {total_reward}, Average reward: {np.mean(total_rewards[-100:])}")

env.close()

plt.plot(total_rewards)
plt.show()

