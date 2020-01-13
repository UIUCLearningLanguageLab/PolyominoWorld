import torch
import torch.nn as nn
import sys
import pickle

class MlNet(nn.Module):
    ############################################################################################################
    def __init__(self, net_type, input_size, hidden_size, output_size, weight_init):

        super(MlNet, self).__init__()
        self.net_type = net_type

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight_init = weight_init

        self.h_x = nn.Linear(input_size, hidden_size).float()
        self.y_h = nn.Linear(hidden_size, output_size).float()
        self.sigmoid = nn.Sigmoid().float()

        self.h_x.apply(self.init_weights)
        self.y_h.apply(self.init_weights)

        self.criterion = nn.MSELoss()
        self.criterion2 = nn.MSELoss(reduction='none')

        self.hidden_states = []

        self.network_states_file = "models/states_{}_i{}-h{}-o{}-ff.csv".format(net_type, input_size, hidden_size,
                                                                                output_size)
        self.network_weights_file = "models/weights_{}_{}-h{}-o{}-ff.csv".format(net_type, input_size, hidden_size,
                                                                                 output_size)

    def forward_item(self, x):
        z_h = self.h_x(x.float())
        h = self.sigmoid(z_h)
        z_o = self.y_h(h)
        o = self.sigmoid(z_o)
        return o, h

    def test_item(self, x, y):
        out, h = self.forward_item(x)
        loss = self.criterion2(out.float(), y.float())
        return out, h, loss

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.uniform_(-self.weight_init, self.weight_init)
            m.bias.data.uniform_(-self.weight_init, self.weight_init)
        else:
            print("Not a linear weight being initialized")
            sys.exit(0)

    def train_item(self, x, y, optimizer):
        out, fc1_out = self.forward_item(x)
        loss = self.criterion(out.float(), y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return out, loss

    def save_network_state(self, x, y, o, h):
        item_state_list = [x, y, h, o]
        self.hidden_states.append(item_state_list)

    def generate_states_file(self):
        outfile = open(self.network_states_file, 'wb')
        pickle.dump(self.hidden_states, outfile)
        outfile.close()

    def generate_weights_file(self):
        outfile = open(self.network_weights_file, 'wb')
        weights_list = [self.h_x, self.y_h]
        pickle.dump(weights_list, outfile)
        outfile.close()


class SlNet(torch.nn.Module):
    def __init__(self, network_file, learning_rate):
        super(SlNet, self).__init__()
        self.network_file = network_file

        self.hidden_state_list = []
        self.x = []
        self.y = []
        self.hidden = []
        self.out = []
        self.h_x = None
        self.y_h = None

        self.load_hidden_states()
        self.input_size = len(self.hidden[0])
        self.output_size = len(self.y[0])
        self.num_items = len(self.y)

        self.y_x = nn.Linear(self.input_size, self.output_size)
        self.sigmoid = nn.Sigmoid().float()
        self.learning_rate = learning_rate

        self.criterion = nn.MSELoss(reduction='none')
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def load_hidden_states(self):
        print("\n\nLoading {}".format(self.network_file))
        infile = open(self.network_file, 'rb')

        state = pickle.load(infile)
        for i in range(len(state)):
            self.x.append(state[i][0])
            self.y.append(state[i][1])
            self.hidden.append(state[i][2])
            self.out.append(state[i][3])

        self.h_x = state[0][4]
        self.y_h = state[0][5]

        print(len(self.x[0]), len(self.y[0]), len(self.hidden[0]), len(self.out[0]))

        infile.close()

    def forward_item(self, x):
        z = self.y_x(x.float())
        o = self.sigmoid(z)
        return o

    def test_item(self, x, y):
        o = self.forward_item(x)
        loss = self.criterion(o.float(), y.float())
        return o, loss

    def train_item(self, x, y):
        o = self.forward_item(x)
        loss = self.criterion(o.float(), y.float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return o, loss
