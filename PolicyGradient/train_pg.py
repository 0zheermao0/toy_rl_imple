import gym
from agent import PolicyGradientAgent
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1')
env = env.unwrapped
episodes = 5500
agent = PolicyGradientAgent(env.observation_space.shape[0], env.action_space.n)
agent.eval_net.to(device)
total_rewards = []
log_probs = []
prg_bar = tqdm(range(episodes))
for episode in prg_bar:
    state = env.reset()
    state = state[0]
    done = False
    total_reward = 0
    while not done:
        env.render()
        action, log_prob = agent.sample(state)
        state_, reward, done, info, _ = env.step(action)
        # if done:
        #     reward = -100
        # else:
        #     reward = 1
        # x, x_dot, theta, theta_dot = state_
        # r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        # reward = r1 + r2
        agent.remember(state, action, reward)
        log_probs.append(log_prob)
        total_reward += reward
        state = state_

    if episode > 0 and episode % 5 == 0:
        agent.learn(torch.stack(log_probs))
        log_probs = []
    # print("Episode: {}, Total Reward: {}".format(episode, round(total_reward, 2)))
    prg_bar.set_description(f"Total Reward: {round(total_reward, 2)}, Avg Reward: {round(np.mean(total_rewards[-100:]), 2)}")
    total_rewards.append(total_reward)

env.close()

print("Average Reward: {}".format())
plt.plot(total_rewards)
plt.show()
