import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PrioritisedReplayBuffer(object):
    def __init__(self, buffer_size, batch_size, alpha, beta, beta_increment):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.max_priority = 1.0
    
    def store(self, transition):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.buffer_size
        self.priorities[self.position] = self.max_priority
    
    def sample(self):
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        s, a, r, s_ = zip(*[self.buffer[i] for i in indices])
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        return s, a, r, s_, indices, weights
    
    def update_priorities(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.priorities[i] = p
        self.max_priority = max(self.max_priority, max(priorities))
    
    def __len__(self):
        return len(self.buffer)

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

class PolicyGradientAgent(object):
    def __init__(self, in_dim, out_dim):
        self.eval_net = PolicyNet(in_dim, 16, out_dim)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.replay_buffer = []
        self.loss_fn = nn.MSELoss()
        self.learn_step_counter = 0
        self.gamma = 0.99

    def sample(self, state):
        state = torch.FloatTensor(state).to(device)
        probs = self.eval_net(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def remember(self, state, action, reward):
        self.replay_buffer.append([state, action, reward])

    def learn(self, log_probs):
        self.learn_step_counter += 1
        rewards = np.array([r for _, _, r in self.replay_buffer])
        for i in range(len(rewards) - 2, -1, -1):
            rewards[i] += rewards[i + 1] * self.gamma
        rewards = torch.from_numpy(rewards).float().to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        loss = (-log_probs * rewards).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.replay_buffer = []

class PPOAgent(object):
    def __init__(self, in_dim, out_dim, K_epochs=4, eps_clip=0.2):
        self.policy_net = PolicyNet(in_dim, 16, out_dim)
        self.policy_old = PolicyNet(in_dim, 16, out_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.replay_buffer = []
        self.loss_fn = nn.MSELoss()
        self.learn_step_counter = 0
        self.gamma = 0.99
        self.k_epochs = K_epochs
        self.eps_clip = eps_clip

        self.policy_old.load_state_dict(self.policy_net.state_dict())

    def sample(self, state):
        state = torch.FloatTensor(state).to(device)
        probs = self.policy_old(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def remember(self, state, action, reward, log_prob):
        self.replay_buffer.append([state, action, reward, log_prob])

    def learn(self):
        self.learn_step_counter += 1
        states = torch.from_numpy(np.array([s for s, _, _, _ in self.replay_buffer])).to(device)
        actions = torch.LongTensor([a for _, a, _, _ in self.replay_buffer]).to(device)
        rewards = np.array([r for _, _, r, _ in self.replay_buffer])
        log_probs = [lp for _, _, _, lp in self.replay_buffer]
        for i in range(len(rewards) - 2, -1, -1):
            rewards[i] += rewards[i + 1] * self.gamma
        rewards = torch.from_numpy(rewards).float().to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        for _ in range(self.k_epochs):
            new_log_probs = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(new_log_probs - torch.FloatTensor(log_probs).to(device))
            surr1 = ratio * rewards
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * rewards
            loss = -torch.min(surr1, surr2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy_net.state_dict())
        self.replay_buffer = []
