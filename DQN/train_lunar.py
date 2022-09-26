import gym
from agent import DQNAgent, DDQNAgent
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make('LunarLander-v2')
env = env.unwrapped
episodes = 1000
dqn = DDQNAgent(env.observation_space.shape[0], env.action_space.n)
dqn.eval_net.to(device)
dqn.target_net.to(device)
total_rewards = []
for episode in range(episodes):
    state = env.reset()
    state = state[0]
    done = False
    total_reward = 0
    while not done:
        env.render()
        action = dqn.sample(state)
        state_, reward, done, info, _ = env.step(action)
        dqn.remember(state, action, reward, state_)
        if len(dqn.replay_buffer) >= dqn.replay_buffer.buffer_size:
            dqn.learn()
        total_reward += reward
        state = state_
    print("Episode: {}, Total Reward: {}".format(episode, round(total_reward, 2)))
    total_rewards.append(total_reward)

env.close()

print("Average Reward: {}".format(np.mean(total_rewards)))
plt.plot(total_rewards)
plt.show()
