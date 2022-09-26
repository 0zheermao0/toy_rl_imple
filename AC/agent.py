import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        res = F.softmax(self.fc3(x), dim=-1)
        return res

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

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        res = self.fc3(x)
        return res

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def __call__(self, x):
        return self.forward(x)

class A2CAgent(object):
    def __init__(self, in_dim, out_dim, proritised=False, K_epochs=4):
        self.proritised = proritised
        self.policy_net = PolicyNet(in_dim, 16, out_dim)
        self.value_net = ValueNet(in_dim, 16, 1)
        self.optim = torch.optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=0.001)
        self.replay_buffer = []
        self.loss_fn = nn.MSELoss()
        self.learn_step_counter = 0
        self.gamma = 0.99
        self.k_epochs = K_epochs

    def sample(self, state):
        state = torch.FloatTensor(state).to(device)
        probs = self.policy_net(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def remember(self, state, action, reward):
        self.replay_buffer.append([state, action, reward])

    def learn(self, log_probs):
        self.learn_step_counter += 1
        states = torch.from_numpy(np.array([s for s, _, _ in self.replay_buffer])).to(device)
        actions = torch.LongTensor([a for _, a, _ in self.replay_buffer]).to(device)
        rewards = np.array([r for _, _, r in self.replay_buffer])
        for i in range(len(rewards) - 2, -1, -1):
            rewards[i] += rewards[i + 1] * self.gamma
        rewards = torch.from_numpy(rewards).float().to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5).detach()
        values = self.value_net(states).squeeze(1)
        advantage = rewards - values
        policy_loss = (-log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        loss = policy_loss + value_loss
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.replay_buffer = []
