import torch
import torch.nn as nn


class FFNet:
    ############################################################################################################
    def __init__(self, input_size, hidden_size, num_classes, learning_rate):
        super(FFNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.L1Loss()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def test(self, out, y):
        loss = self.criterion(out, y)
        return loss

    def train(self, num_epochs, x, y):
        for e in range(num_epochs):
            epoch_loss = 0
            for i in range(4):
                out = self.forward(x[i])
                loss = self.criterion(out, y[i])
                epoch_loss += loss.detach().numpy()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if e % 10 == 0:
                print(e, epoch_loss)


