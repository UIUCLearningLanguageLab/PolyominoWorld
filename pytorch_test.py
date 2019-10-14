import torch
import numpy as np
from src.networks import pytorch_nets

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

model = pytorch_nets.FFNet(input_size, hidden_size, output_size, weight_init, learning_rate)

for e in range(num_epochs):
    epoch_loss = 0
    for i in range(4):
        out, loss = model.train_item(x[i], y[i])
        epoch_loss += loss
    if e % 100 == 0:
        print(e, epoch_loss)

for i in range(4):
    out = model.forward_item(x[i])
    print("{} {} {}".format(x[i].detach().numpy(), y[i].detach().numpy(), out[-1].detach().numpy()))
