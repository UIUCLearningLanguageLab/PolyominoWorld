import torch
import torch.nn as nn


class FFNet(nn.Module):
    ############################################################################################################
    def __init__(self, input_size, hidden_size, output_size, weight_init, learning_rate):

        super(FFNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

        self.learning_rate = learning_rate

        self.weight_init = weight_init

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        self.fc1.apply(self.init_weights)
        self.fc2.apply(self.init_weights)

    def forward_item(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def test_item(self, x, y):
        out = self.forward_item(x)
        loss = self.criterion(out, y)
        return out, loss

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.uniform_(-self.weight_init, self.weight_init)
            m.bias.data.uniform_(-self.weight_init, self.weight_init)

    def train_item(self, x, y):
        out = self.forward_item(x)
        loss = self.criterion(out, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return out, loss