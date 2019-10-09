import torch
import torch.nn as nn
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, weight_init):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.weight_init = weight_init
        self.apply(self.init_weights)

    def forward(self, x):
        z_h = self.fc1(x)
        h = self.sigmoid(z_h)
        z_o = self.fc2(h)
        o = self.sigmoid(z_o)
        return z_h, h, z_o, o

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.uniform_(-self.weight_init, self.weight_init)
            m.bias.data.uniform_(-self.weight_init, self.weight_init)

# AND Dataset
x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], requires_grad=True)
y = torch.tensor([[0.], [1.], [1.], [0.]], requires_grad=True)

# Hyper-parameters
input_size = 2
hidden_size = 2
output_size = 1
num_epochs = 5000
learning_rate = 0.3
weight_init = 1

np.set_printoptions(precision=4, suppress=True)

model = NeuralNet(input_size, hidden_size, output_size, weight_init)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for e in range(num_epochs):
    epoch_loss = 0
    for i in range(4):
        out = model.forward(x[i])
        loss = criterion(out[-1], y[i])
        epoch_loss += loss.detach().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if e % 100 == 0:
        print(e, epoch_loss)

for i in range(4):
    out = model.forward(x[i])
    print("{} {} {}".format(x[i].detach().numpy(), y[i].detach().numpy(), out[-1].detach().numpy()))
