import gym

env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
env.reset()
total_rewards = []
for _ in range(1000):
    done = False
    total_reward = 0
    while not done:
        _, r, done, _, _ = env.step(env.action_space.sample())
        total_reward += r
    total_rewards.append(total_reward)
    print('Average reward: {}'.format(sum(total_rewards) / len(total_rewards)))
    env.reset()
env.close()
