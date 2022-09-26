import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tianshou as ts
from torch.utils.tensorboard import SummaryWriter

task = 'ALE/SpaceInvaders-v5'
lr, epoch, batch_size = 1e-3, 1, 64
train_num, test_num = 10, 100
gamma, n_step, target_freq = 0.99, 3, 320
buffer_size = 20000
eps_train, eps_test = 0.1, 0.05
step_per_epoch, step_per_collect = 10000, 10
logger = ts.utils.TensorboardLogger(SummaryWriter('log/dqn'))

train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task, render_mode='rgb_array') for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task, render_mode='rgb_array') for _ in range(test_num)])

class ConvNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConvNet, self).__init__()
        input_size = (3, 210, 160)
        self.input_size = input_size
        self.output_size = output_size

        self.conv = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )

    def feature_size(self):
        return self.conv(torch.zeros(1, *self.input_size)).view(1, -1).size(1)

    def forward(self, x, state=None, info={}):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.permute(0, 3, 1, 2)
        batch = x.size(0)
        x = self.conv(x / 255.)
        x = x.reshape(batch, -1)
        return self.fc(x), state

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

env = gym.make(task, render_mode='human')
net = ConvNet(env.observation_space.shape, env.action_space.n)
optim = torch.optim.Adam(net.parameters(), lr=lr)
policy = ts.policy.DQNPolicy(net, optim, discount_factor=gamma, estimation_step=n_step, target_update_freq=target_freq)
policy.load_state_dict(torch.load('policy.pth'))
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect, test_num, batch_size,
    update_per_step = 1/step_per_collect, 
    train_fn = lambda epoch, env_step: policy.set_eps(eps_train),
    test_fn = lambda epoch, env_step: policy.set_eps(eps_test),
    logger = logger,
)

torch.save(policy.state_dict(), 'policy.pth')
policy.load_state_dict(torch.load('policy.pth'))

policy.eval()
policy.set_eps(eps_test)
collector = ts.data.Collector(policy, env, exploration_noise=True)
collector.collect(n_episode=1, render=1/35)

