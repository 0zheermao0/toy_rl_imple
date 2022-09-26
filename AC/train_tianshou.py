import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tianshou as ts
from torch.utils.tensorboard import SummaryWriter

task = 'CartPole-v1'
lr, epoch, batch_size = 1e-3, 1, 64
train_num, test_num = 10, 100
gamma, n_step, target_freq = 0.99, 3, 320
buffer_size = 20000
eps_train, eps_test = 0.1, 0.05
step_per_epoch, step_per_collect = 10000, 10
logger = ts.utils.TensorboardLogger(SummaryWriter('log/a2c'))

train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(test_num)])

class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, state=None, info={}):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        res = F.softmax(self.fc3(x), dim=-1)
        return res, state

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def __call__(self, x):
        return self.forward(x)

class ValueNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ValueNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, state=None, info={}):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        res = self.fc3(x)
        return res, state

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def __call__(self, x):
        return self.forward(x)

env = gym.make(task)
policy_net = PolicyNet(env.observation_space.shape[0], 16, env.action_space.n)
value_net = ValueNet(env.observation_space.shape[0], 16, 1)
optim = torch.optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=lr)
policy = ts.policy.A2CPolicy(policy_net, value_net, optim, dist_fn=F.softmax)
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

