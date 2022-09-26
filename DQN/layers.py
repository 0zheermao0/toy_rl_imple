import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True, training=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.training = training
        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)
        self.reset_parameters()
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        if bias:
            self.register_buffer('epsilon_bias', torch.zeros(out_features))
        else:
            self.register_buffer('epsilon_bias', None)
        self.reset_noise()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight_mu.data.uniform_(-std, std)
        self.weight_sigma.data.fill_(self.sigma_init)
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-std, std)
            self.bias_sigma.data.fill_(self.sigma_init)
        
    def reset_noise(self):
        self.epsilon_weight.data.normal_()
        if self.bias_mu is not None:
            self.epsilon_bias.data.normal_()

    def forward(self, x):
        self.reset_noise()
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.epsilon_weight))
            if self.bias_mu is not None:
                bias = self.bias_mu + self.bias_sigma.mul(Variable(self.epsilon_bias))
        else:
            weight = self.weight_mu
            if self.bias_mu is not None:
                bias = self.bias_mu
        return F.linear(x, weight, bias)

class NoisyFactorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.4, bias=True, training=True):
        super(NoisyFactorizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.training = training
        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)
        self.reset_parameters()
        self.register_buffer('epsilon_input', torch.zeros(in_features))
        self.register_buffer('epsilon_output', torch.zeros(out_features))
        self.reset_noise()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight_mu.data.uniform_(-std, std)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-std, std)
            self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))
        
    def reset_noise(self):
        eps_i = torch.randn(self.in_features)
        eps_o = torch.randn(self.out_features)
        self.epsilon_input = eps_i.sign() * eps_i.abs().sqrt()
        self.epsilon_output = eps_o.sign() * eps_o.abs().sqrt()

    def forward(self, x):
        self.reset_noise()
        if self.training:
            epsilon_weight = self.epsilon_output.outer(self.epsilon_input).to(x.device)
            epsilon_bias = self.epsilon_output.to(x.device)
            weight = self.weight_mu + self.weight_sigma.mul(Variable(epsilon_weight))
            if self.bias_mu is not None:
                bias = self.bias_mu + self.bias_sigma.mul(Variable(epsilon_bias))
        else:
            weight = self.weight_mu
        return F.linear(x, weight, bias)
