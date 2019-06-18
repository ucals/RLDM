import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, in_channels, hidden, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features=in_channels, out_features=hidden)
        self.fc2 = nn.Linear(in_features=hidden, out_features=hidden)
        self.fc3 = nn.Linear(in_features=hidden, out_features=num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DuelingDQN(nn.Module):
    def __init__(self, in_channels, hidden, num_actions):
        super(DuelingDQN, self).__init__()
        self.num_actions = num_actions
        self.fc1 = nn.Linear(in_features=in_channels, out_features=hidden)
        self.fc2 = nn.Linear(in_features=hidden, out_features=hidden)
        self.fc2_adv = nn.Linear(in_features=hidden, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=hidden, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_len = x.size(0)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        adv = self.fc2_adv(x)
        val = self.fc2_val(x).expand(batch_len, self.num_actions)
        x = val + adv - adv.mean(1).unsqueeze(1).expand(batch_len, self.num_actions)
        return x