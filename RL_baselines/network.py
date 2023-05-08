import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    def __init__(self, state_dims, layer1_dims, layer2_dims, action_dim, max_action, mode):
        super(ActorNetwork, self).__init__()
        if mode == "NAVTD3":
            pass
        elif mode == "TD3":
            pass
        self.mode = mode
        self.action_before_tanh = [0.0, 0.0]
        self.layer_activation = nn.LeakyReLU()
        self.out_activation = nn.Tanh()

        self.max_action = max_action
        self.first = nn.Linear(state_dims, layer1_dims)
        self.second = nn.Linear(layer1_dims, layer2_dims)
        self.third = nn.Linear(layer2_dims, action_dim)
        print(f"actor parameter count: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, state):
        # Layer 1
        out = self.first(state)
        out = self.layer_activation(out)

        out = self.second(out)
        out = self.layer_activation(out)

        before_tanh = self.third(out)

        if self.mode == "NAVTD3":
            if out.shape[0] == 1:
                self.action_before_tanh[0], self.action_before_tanh[1] = before_tanh[0, 0], before_tanh[0, 1]
        o1 = self.out_activation(before_tanh) * self.max_action

        return o1, before_tanh


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dims, layer1_dims, layer2_dims):
        super(CriticNetwork, self).__init__()
        self.activation = nn.LeakyReLU()

        self.first = nn.Linear(state_dim + action_dims, layer1_dims)
        self.second = nn.Linear(layer1_dims + action_dims, layer2_dims)
        self.third = nn.Linear(layer2_dims + action_dims*2, 1)
        print(f"critic parameter count: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, state, action):
        out = self.first(torch.cat((state, action), dim=1))
        out = self.activation(out)

        out = self.second(torch.cat((out, action), dim=1))
        out = self.activation(out)

        out = self.third(torch.cat((out, action, torch.abs(action)), dim=1))
        return out
