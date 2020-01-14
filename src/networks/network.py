import torch
import torch.nn as nn
import sys
import pickle
import datetime
import os


class MlNet(nn.Module):
    ############################################################################################################
    def __init__(self, net_type, training_set, hidden_size, learning_rate, weight_init):

        super(MlNet, self).__init__()
        self.net_name = None
        self.net_type = net_type
        self.training_set = training_set
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_init = weight_init

        self.input_size = training_set.world_size

        if net_type == 'autoassociator':
            self.output_size = training_set.world_size
        elif net_type == 'classifier':
            self.output_size = training_set.num_included_features
        else:
            print("Net Type {} not recognized")
            sys.exit()

        self.current_epoch = 0

        self.h_x = nn.Linear(self.input_size, self.hidden_size).float()
        self.y_h = nn.Linear(self.hidden_size, self.output_size).float()
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
        o, h = self.forward_item(x)
        loss = self.criterion(o.float(), y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return o, loss

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
            network_state_list.append((dataset.x[i], dataset.y[i], o.detach().numpy(), h.detach().numpy()))

        file_location = "models/" + self.net_name + "/states_e{}.csv".format(self.current_epoch)
        outfile = open(file_location, 'wb')
        pickle.dump(network_state_list, outfile)
        outfile.close()

    def save_network_weights(self):
        file_location = "models/" + self.net_name + "/weights_e{}.csv".format(self.current_epoch)
        outfile = open(file_location, 'wb')
        weights_list = [self.h_x, self.y_h]
        pickle.dump(weights_list, outfile)
        outfile.close()


class SlNet(torch.nn.Module):
    def __init__(self, training_set, learning_rate, weight_init):
        super(SlNet, self).__init__()
        self.training_set = training_set
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.net_name = None
        self.net_type = "classifier"

        self.current_epoch = 0

        self.input_size = training_set.h_size
        self.output_size = training_set.num_included_features

        self.y_x = nn.Linear(self.output_size, self.input_size).float()
        self.sigmoid = nn.Sigmoid().float()

        self.y_x.apply(self.init_weights)

        self.criterion = nn.MSELoss()
        self.criterion2 = nn.MSELoss(reduction='none')

        self.hidden_states = []

        self.start_datetime = datetime.datetime.timetuple(datetime.datetime.now())
        self.create_network_directory()

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.uniform_(-self.weight_init, self.weight_init)
            m.bias.data.uniform_(-self.weight_init, self.weight_init)
        else:
            print("Not a linear weight being initialized")
            sys.exit(0)

    def forward_item(self, x):
        z_o = self.y_x(x.float())
        o = self.sigmoid(z_o)
        return o

    def test_item(self, x, y):
        o = self.forward_item(x)
        loss = self.criterion2(o.float(), y.float())
        return o, loss

    def train_item(self, x, y, optimizer):
        o = self.forward_item(x)
        loss = self.criterion(o.float(), y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return o, loss

    def save_network_weights(self):
        file_location = "models/" + self.net_name + "/weights_e{}.csv".format(self.current_epoch)
        outfile = open(file_location, 'wb')
        weights_list = [self.y_x]
        pickle.dump(weights_list, outfile)
        outfile.close()

    def create_network_directory(self):
        self.net_name = "{}_{}_{}_{}_{}_{}_{}".format(self.net_type+"B",
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
        f.write("output_size: {}\n".format(self.output_size))
        f.write("learning_rate: {}\n".format(self.learning_rate))
        f.write("weight_init: {}\n".format(self.weight_init))
        f.write("training_file: {}".format(self.training_set.world_state_filename))
        f.close()