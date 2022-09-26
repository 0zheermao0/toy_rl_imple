import gym
from agent import DQNAgent, DDQNAgent
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1')
env = env.unwrapped
episodes = 250
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
agent.eval_net.to(device)
agent.target_net.to(device)
total_rewards = []
pgr_bar = tqdm(range(episodes))
for episode in pgr_bar:
    state = env.reset()
    state = state[0]
    done = False
    total_reward = 0
    while not done:
        env.render()
        action = agent.sample(state)
        state_, reward, done, info, _ = env.step(action)
        x, x_dot, theta, theta_dot = state_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2
        # if x < -2.4 or x > 2.4 or theta < -0.2094 or theta > 0.2094:
        #     reward = -1
        # else:
        #     reward = 1
        agent.remember(state, action, reward, state_)
        if len(agent.replay_buffer) >= agent.replay_buffer_size:
            agent.learn()
        total_reward += reward
        state = state_
    pgr_bar.set_description("Total Reward: {}, Average Reward:{}".format(round(total_reward, 2), round(np.mean(total_rewards[-10:]), 2)))
    total_rewards.append(total_reward)

env.close()

print("Average Reward: {}".format(np.mean(total_rewards)))
plt.plot(total_rewards)
plt.show()
