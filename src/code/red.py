class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

        self.layer_norm1 = nn.LayerNorm(128)
        self.layer_norm2 = nn.LayerNorm(64)

    def forward(self, x):
        x = torch.relu(self.layer_norm1(self.fc1(x)))
        x = torch.relu(self.layer_norm2(self.fc2(x)))
        return self.fc3(x)
