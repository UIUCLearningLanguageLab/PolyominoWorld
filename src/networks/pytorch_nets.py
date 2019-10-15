import torch
import torch.nn as nn
import sys


class FFNet(nn.Module):
    ############################################################################################################
    def __init__(self, input_size, hidden_size, output_size, weight_init, learning_rate):

        super(FFNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

        self.learning_rate = learning_rate

        self.weight_init = weight_init

        self.fc1.apply(self.init_weights)
        self.fc2.apply(self.init_weights)

        self.criterion = nn.MSELoss()

    def forward_item(self, x):
        z_h = self.fc1(x)
        h = self.sigmoid(z_h)
        z_o = self.fc2(h)
        o = self.sigmoid(z_o)
        return o

    def test_item(self, x, y):
        out = self.forward_item(x)
        loss = self.criterion(out, y)
        return out, loss

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.uniform_(-self.weight_init, self.weight_init)
            m.bias.data.uniform_(-self.weight_init, self.weight_init)
        else:
            print("Not a linear weight being initialized")
            sys.exit(0)

    def train_item(self, x, y, optimizer):
        out = self.forward_item(x)
        loss = self.criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return out, loss