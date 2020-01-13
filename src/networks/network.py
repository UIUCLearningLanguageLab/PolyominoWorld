import torch
import torch.nn as nn
import sys
import pickle
import datetime
import os


class MlNet(nn.Module):
    ############################################################################################################
    def __init__(self, net_type, training_set, hidden_size, output_size, learning_rate, weight_init):

        super(MlNet, self).__init__()
        self.net_name = None
        self.net_type = net_type

        self.training_set = training_set
        self.input_size = training_set.world_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.current_epoch = 0

        self.h_x = nn.Linear(self.input_size, hidden_size).float()
        self.y_h = nn.Linear(hidden_size, output_size).float()
        self.sigmoid = nn.Sigmoid().float()

        self.h_x.apply(self.init_weights)
        self.y_h.apply(self.init_weights)

        self.criterion = nn.MSELoss()
        self.criterion2 = nn.MSELoss(reduction='none')

        self.hidden_states = []

        self.start_datetime = datetime.datetime.timetuple(datetime.datetime.now())
        self.create_network_directory()

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

    def create_network_directory(self):
        self.net_name = "{}_{}_{}_{}_{}_{}_{}".format(self.net_type,
                                                   self.start_datetime[0],
                                                   self.start_datetime[1],
                                                   self.start_datetime[2],
                                                   self.start_datetime[3],
                                                   self.start_datetime[4],
                                                   self.start_datetime[5],
                                                   self.start_datetime[6])
        try:
            os.mkdir("models/" + self.net_name)
        except:
            print("ERROR: Network {} directory already exists".format(self.net_name))
            sys.exit()
        file_location = "models/" + self.net_name + "/network_properties.csv"
        f = open(file_location, 'w')
        f.write("network_name: {}\n".format(self.net_name))
        f.write("network_type: {}\n".format(self.net_type))
        f.write("input_size: {}\n".format(self.input_size))
        f.write("hidden_size: {}\n".format(self.hidden_size))
        f.write("output_size: {}\n".format(self.output_size))
        f.write("learning_rate: {}\n".format(self.learning_rate))
        f.write("weight_init: {}\n".format(self.weight_init))
        f.write("training_file: {}".format(self.training_set.world_state_filename))
        f.close()

    def save_network_states(self, dataset, x_type, y_type):
        network_state_list = []

        dataset.create_xy(x_type, y_type, False, False)

        for i in range(len(dataset.x)):
            o, h, o_cost = self.test_item(dataset.x[i], dataset.y[i])
            network_state_list.append((dataset.x[i], dataset.y[i], o, h))

        file_location = "models/" + self.net_name + "/states_e{}.csv".format(self.current_epoch)
        outfile = open(file_location, 'wb')
        pickle.dump(self.hidden_states, outfile)
        outfile.close()

    def save_network_weights(self):
        file_location = "models/" + self.net_name + "/weights_e{}.csv".format(self.current_epoch)
        outfile = open(file_location, 'wb')
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
