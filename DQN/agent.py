import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.distributions import Categorical
from layers import NoisyFactorizedLinear
from torch.autograd import Variable

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

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dueling=False, noisy=False):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dueling = dueling

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

        if self.dueling:
            self.fc3 = nn.Linear(self.hidden_size, 1)

        if noisy:
            self.fc1 = NoisyFactorizedLinear(self.input_size, self.hidden_size)
            self.fc2 = NoisyFactorizedLinear(self.hidden_size, self.output_size)

            if self.dueling:
                self.fc3 = NoisyFactorizedLinear(self.hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        res = self.fc2(x)

        if self.dueling:
            value = self.fc3(x)
            res = res - res.mean() + value
        return res

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def __call__(self, x):
        return self.forward(x)

class ConvNet(nn.Module):
    def __init__(self, input_size, output_size, dueling=False, noisy=False):
        super(ConvNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dueling = dueling

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_size[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(self.feature_size(), 512)
        self.fc2 = nn.Linear(512, self.output_size)

        if self.dueling:
            self.fc3 = nn.Linear(512, 1)

        if noisy:
            self.fc1 = NoisyFactorizedLinear(self.feature_size(), 512)
            self.fc2 = NoisyFactorizedLinear(512, self.output_size)

            if self.dueling:
                self.fc3 = NoisyFactorizedLinear(512, 1)

    def feature_size(self):
        return self.conv(Variable(torch.zeros(1, *self.input_size))).view(1, -1).size(1)

    def forward(self, x):
        features = self.conv(x).reshape(x.size(0), -1)
        x = F.relu(self.fc1(features))
        res = self.fc2(x)

        if self.dueling:
            value = self.fc3(x)
            res = res - res.mean() + value
        return res

class DQNAgent(object):
    def __init__(self, in_dim, out_dim, dueling=False, noisy=False, use_conv=False):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.replay_buffer = []
        self.replay_buffer_size = 2000
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.noisy = noisy
        self.use_conv = use_conv
        self.batch_size = 128

        if self.use_conv:
            self.eval_net = ConvNet(in_dim, out_dim, dueling, noisy)
            self.target_net = ConvNet(in_dim, out_dim, dueling, noisy)
        else:
            self.eval_net = Net(in_dim, 50, out_dim, dueling, noisy)
            self.target_net = Net(in_dim, 50, out_dim, dueling, noisy)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def sample(self, state, epsilon):
        state = torch.unsqueeze(torch.from_numpy(np.array(state)).float(), 0).permute(0, 3, 1, 2).to(device)
        if self.noisy:
            action = self.eval_net(state).max(1)[1].data.cpu().numpy()[0]
        else:
            if np.random.uniform() < epsilon:
                action = np.random.randint(0, self.out_dim)
            else:
                q_value = self.eval_net(state)
                action = torch.max(q_value, 1)[1].data.cpu().numpy()[0]
        return action

    def save(self, path):
        agent_dict = {
            'net': self.eval_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(agent_dict, path)

    def load(self, path):
        agent_dict = torch.load(path)
        self.eval_net.load_state_dict(agent_dict['net'])
        self.target_net.load_state_dict(agent_dict['net'])
        self.optimizer.load_state_dict(agent_dict['optimizer'])

    def remember(self, state, action, reward, next_state):
        transition = (state, action, reward, next_state)
        if self.memory_counter < self.replay_buffer_size:
            self.replay_buffer.append(transition)
            self.memory_counter += 1
        else:
            self.replay_buffer[self.memory_counter % self.replay_buffer_size] = transition
            self.memory_counter += 1

    # def learn(self):
    #     if self.learn_step_counter % 100 == 0:
    #         self.target_net.load_state_dict(self.eval_net.state_dict())
    #     self.learn_step_counter += 1
    #     sample_index = np.random.choice(self.replay_buffer_size, 32)
    #     batch_memory = np.array(self.replay_buffer)[sample_index, :]    
    #     batch_state = torch.FloatTensor(batch_memory[:, :self.eval_net.input_size]).to(device)
    #     batch_action = torch.LongTensor(batch_memory[:, self.eval_net.input_size:self.eval_net.input_size+1].astype(int)).to(device)
    #     batch_reward = torch.FloatTensor(batch_memory[:, self.eval_net.input_size+1:self.eval_net.input_size+2]).to(device)
    #     batch_state_ = torch.FloatTensor(batch_memory[:, -self.eval_net.input_size:]).to(device)
    #     print('\nstates: ', batch_state, '\nactions: ', batch_action, '\nrewards: ', batch_reward, '\nstates_: ', batch_state_)

    #     q_eval = self.eval_net(batch_state).gather(1, batch_action)
    #     q_next = self.target_net(batch_state_).detach()
    #     q_target = batch_reward + 0.9 * q_next.max(1)[0].view(32, 1)
    #     loss = self.loss_fn(q_eval, q_target)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    def learn(self):
        if self.learn_step_counter % 100 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        sample_index = np.random.choice(self.replay_buffer_size, self.batch_size)

        states, actions, rewards, states_ = zip(*[self.replay_buffer[i] for i in sample_index])
        states = torch.from_numpy(np.array(states)).float().to(device).view(self.batch_size, 4, 84, 84)
        actions = torch.LongTensor(actions).to(device).view(self.batch_size, -1)
        rewards = torch.FloatTensor(rewards).to(device).view(self.batch_size, -1)
        states_ = torch.from_numpy(np.array(states_)).float().to(device).view(self.batch_size, 4, 84, 84)

        q_eval = self.eval_net(states).gather(1, actions)
        # 根据target net获取下一个state的最大Q值
        q_next = self.target_net(states_).detach()
        q_target = rewards + 0.9 * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class DDQNAgent(object):
    def __init__(self, in_dim, out_dim, dueling=False):
        self.eval_net = Net(in_dim, 50, out_dim, dueling)
        self.target_net = Net(in_dim, 50, out_dim, dueling)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.replay_buffer = PrioritisedReplayBuffer(2000, 32, 0.6, 0.4, 0.001)
        self.memory_counter = 0

    def sample(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
        if np.random.uniform() < 0.1:
            action = np.random.randint(0, 2)
        else:
            q_value = self.eval_net(state)
            action = torch.max(q_value, 1)[1].data.cpu().numpy()[0]
        return action

    def save(self, path):
        agent_dict = {
            'net': self.eval_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(agent_dict, path)

    def load(self, path):
        agent_dict = torch.load(path)
        self.eval_net.load_state_dict(agent_dict['net'])
        self.optimizer.load_state_dict(agent_dict['optimizer'])

    def remember(self, state, action, reward, next_state):
        # transition = np.hstack((state, [action, reward], next_state))
        transition = (state, action, reward, next_state)
        self.replay_buffer.store(transition)

    def learn(self):
        if self.learn_step_counter % 100 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        states, actions, rewards, states_, indices, weights = self.replay_buffer.sample()
        states = torch.FloatTensor(states).to(device).view(32, -1)
        actions = torch.LongTensor(actions).to(device).view(32, -1)
        rewards = torch.FloatTensor(rewards).to(device).view(32, -1)
        states_ = torch.FloatTensor(states_).to(device).view(32, -1)

        q_eval = self.eval_net(states).gather(1, actions)
        # eval net输出Q值的最大值的index，也是最大值对应的action
        q_eval_next = self.eval_net(states_).max(1)[1].view(32, 1)
        # 根据eval net采取的action，从target net中取出对应的q值
        q_next = self.target_net(states_).detach().gather(1, q_eval_next)
        q_target = rewards + 0.9 * q_next
        abs_errors = Variable(torch.abs(q_eval - q_target) * torch.FloatTensor(weights).to(device).view(32, 1), requires_grad=True)
        loss = self.loss_fn(q_eval, q_target) + torch.mean(abs_errors)
        self.replay_buffer.update_priorities(indices, abs_errors.data.cpu().numpy())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
