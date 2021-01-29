import torch
import torch.nn as nn
import sys
import pickle
import datetime
import os
from polyomino_world.networks import dataset


class MlNet(nn.Module):
    ############################################################################################################
    def __init__(self):

        super(MlNet, self).__init__()
        self.net_name = None
        self.x_type = None
        self.y_type = None
        self.training_set = None
        self.optimizer = None
        self.hidden_size = None
        self.learning_rate = None
        self.weight_init = None
        self.processor = None
        self.input_size = None
        self.output_size = None
        self.current_epoch = None
        self.h_x = None
        self.y_h = None
        self.sigmoid = None
        self.criterion = None
        self.criterion2 = None
        self.start_datetime = None
        self.model_directory = None
        self.performance_list = None
        self.training_time = None
        self.hidden_actf = None

        self.criterion = nn.MSELoss()
        self.criterion2 = nn.MSELoss(reduction='none')
        self.sigmoid = nn.Sigmoid().float()
        self.tanh = nn.Tanh().float()
        self.relu = nn.ReLU().float()

    def init_model(self, x_type, y_type, training_set,
                   hidden_size, hidden_actf, optimizer, learning_rate, weight_init, processor):

        self.x_type = x_type
        self.y_type = y_type
        self.training_set = training_set
        self.optimizer = optimizer
        self.hidden_size = hidden_size
        self.hidden_actf = hidden_actf
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.processor = processor
        self.input_size = training_set.world_size

        if self.y_type == 'WorldState':
            self.output_size = training_set.world_size
        elif self.y_type == 'FeatureVector':
            self.output_size = training_set.num_included_features
        else:
            print("Y Type {} not recognized")
            sys.exit()

        self.current_epoch = 0

        self.h_x = nn.Linear(self.input_size, self.hidden_size).float()
        self.y_h = nn.Linear(self.hidden_size, self.output_size).float()

        self.h_x.apply(self.init_weights)
        self.y_h.apply(self.init_weights)

        self.start_datetime = datetime.datetime.timetuple(datetime.datetime.now())

        self.training_time = 0

        self.create_network_directory()

        if self.processor == 'GPU':
            if torch.cuda.is_available():
                self.cuda(0)

    def load_model(self, model_directory, included_features, processor):
        self.net_name = model_directory
        params_file = open('models/' + model_directory + '/network_properties.csv')
        for line in params_file:

            data = (line.strip().strip('\n').strip()).split()
            parameter = data[0]
            value = data[1]

            if parameter == 'x_type:':
                self.x_type = value
            elif parameter == 'y_type:':
                self.y_type = value
            elif parameter == 'input_size:':
                self.input_size = int(value)
            elif parameter == 'hidden_size:':
                self.hidden_size = int(value)
            elif parameter == 'hidden_actf:':
                self.hidden_actf = value
            elif parameter == 'output_size:':
                self.output_size = int(value)
            elif parameter == 'optimizer:':
                self.optimizer = value
            elif parameter == 'learning_rate:':
                self.learning_rate = float(value)
            elif parameter == 'weight_init:':
                self.weight_init = float(value)
            elif parameter == 'current_epoch:':
                self.current_epoch = int(value)
            elif parameter == 'training_time:':
                self.training_time = float(value)
            elif parameter == 'training_set:':
                self.training_set = dataset.DataSet(value, None, included_features, processor)
        params_file.close()

        weight_file = "models/" + self.net_name + "/weights/epoch {}.csv".format(self.current_epoch)
        weight_file = open(weight_file, 'rb')
        weights_list = pickle.load(weight_file)
        weight_file.close()

        self.h_x = weights_list[0]
        self.y_h = weights_list[1]

    def forward_item(self, x):
        z_h = self.h_x(x.float())
        if self.hidden_actf == 'tanh':
            h = self.tanh(z_h)
        elif self.hidden_actf == 'sigmoid':
            h = self.tanh(z_h)
        elif self.hidden_actf == 'relu':
            h = self.relu(z_h)
        else:
            print("ERROR: Improper hidden activation function")
            raise RuntimeError
        z_o = self.y_h(h)

        if self.y_type == 'FeatureVector':
            o = self.sigmoid(z_o)
        elif self.y_type == 'WorldState':
            o = self.tanh(z_o)
        else:
            print("ERROR: y-type not recognized")
            sys.exit()
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
        try:
            print("Creating models directory")
            os.mkdir('models')
        except Exception as e:
            pass

        if self.x_type == 'WorldState':
            x_type = "WS"
        elif self.x_type == 'HiddenState':
            x_type = 'HS'
        else:
            print("X Type not recognized in directory creation")
            sys.exit()

        if self.y_type == 'WorldState':
            y_type = "WS"
        elif self.y_type == 'FeatureVector':
            y_type = 'FV'
        else:
            print("Y Type not recognized in directory creation")
            sys.exit()

        self.net_name = "{}_{}_{}_{}_{}_{}_{}_{}_top_bottom_train_top_test_full_first_stage_check".format(x_type, y_type,
                                                         self.start_datetime[0],
                                                         self.start_datetime[1],
                                                         self.start_datetime[2],
                                                         self.start_datetime[3],
                                                         self.start_datetime[4],
                                                         self.start_datetime[5],
                                                         self.start_datetime[6])
        try:
            os.mkdir("models/" + self.net_name)
        except Exception as e:
            print(e)
            sys.exit()

        header_string = "epoch,time,training_cost,test_cost"
        for i in range(self.training_set.num_included_feature_types):
            header_string += ',{}_training_cost'.format(self.training_set.included_feature_type_list[i])
        for i in range(self.training_set.num_included_feature_types):
            header_string += ',{}_test_cost'.format(self.training_set.included_feature_type_list[i])
        for i in range(self.training_set.num_included_feature_types):
            header_string += ',{}_training_accuracy'.format(self.training_set.included_feature_type_list[i])
        for i in range(self.training_set.num_included_feature_types):
            header_string += ',{}_test_accuracy'.format(self.training_set.included_feature_type_list[i])
        for i in range(self.training_set.num_included_features):
            header_string += ',{}_training_accuracy'.format(self.training_set.included_feature_list[i])
        for i in range(self.training_set.num_included_features):
            header_string += ',{}_test_accuracy'.format(self.training_set.included_feature_list[i])

        f = open("models/" + self.net_name + "/performance.csv", 'w')
        f.write(header_string + "\n")
        f.close()

        self.save_network_properties()

    def save_network_properties(self):
        file_location = "models/" + self.net_name + "/network_properties.csv"
        f = open(file_location, 'w')
        f.write("network_name: {}\n".format(self.net_name))
        f.write("x_type: {}\n".format(self.x_type))
        f.write("y_type: {}\n".format(self.y_type))
        f.write("input_size: {}\n".format(self.input_size))
        f.write("hidden_size: {}\n".format(self.hidden_size))
        f.write("hidden_actf: {}\n".format(self.hidden_actf))
        f.write("output_size: {}\n".format(self.output_size))
        f.write("optimizer: {}\n".format(self.optimizer))
        f.write("learning_rate: {}\n".format(self.learning_rate))
        f.write("weight_init: {}\n".format(self.weight_init))
        f.write("training_set: {}\n".format(self.training_set.world_state_filename))
        # may need adding testing set
        f.write("current_epoch: {}\n".format(self.current_epoch))
        f.write("training_time: {}".format(self.training_time))
        f.close()

    def save_network_states(self, dataset):
        network_state_list = []
        dataset.create_xy(self, False, False)
        for i in range(len(dataset.x)):
            o, h, o_cost = self.test_item(dataset.x[i], dataset.y[i])
            network_state_list.append((dataset.x[i], dataset.y[i], o.detach().cpu().numpy(), h.detach().cpu().numpy()))
        file_location = "models/" + self.net_name + "/states.csv".format(self.current_epoch)
        outfile = open(file_location, 'wb')
        pickle.dump(network_state_list, outfile)
        outfile.close()

    # def save_network_weights(self):
    #     file_location = "models/" + self.net_name + "/weights.csv".format(self.current_epoch)
    #     outfile = open(file_location, 'wb')
    #     weights_list = [self.h_x, self.y_h]
    #     pickle.dump(weights_list, outfile)
    #     outfile.close()

    def save_network_weights(self, current_epoch):
        folder_name = "models/" + self.net_name + "/weights"
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        file_name = 'epoch {}.csv'.format(current_epoch)
        # file = os.path.join(folder_name, file_name)
        # os.makedirs(file)
        file = os.path.expanduser(folder_name+'/'+file_name)
        with open(file, 'wb') as f:
            weights_list = [self.h_x, self.y_h]
            pickle.dump(weights_list, f)
        f.close()


    def save_network_performance(self):
        file_location = "models/" + self.net_name + "/performance.csv"
        output_string = ""
        for item in self.performance_list:
            output_string += str(item) + ","
        output_string = output_string[:-1]
        f = open(file_location, 'a')
        f.write(output_string + "\n")
        f.close()


class SlNet(torch.nn.Module):
    def __init__(self, x_type, y_type, training_set, learning_rate, weight_init):
        super(SlNet, self).__init__()

        self.x_type = x_type
        self.y_type = y_type
        self.training_set = training_set
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.net_name = None

        self.current_epoch = 0

        if x_type == 'world_state':
            self.input_size = training_set.world_size
        elif x_type == 'hidden_state':
            self.input_size = training_set.h_size
        else:
            print("X Type {} not recognized".format(self.x_type))
            sys.exit()

        if y_type == 'world_state':
            self.output_size = training_set.world_size
        elif y_type == 'feature_vector':
            self.output_size = training_set.num_included_features
        else:
            print("Y Type {} not recognized".format(self.y_type))
            sys.exit()

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
        self.net_name = "{}_{}_{}_{}_{}_{}_{}_{}".format(self.x_type, self.y_type,
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
        f.write("x_type: {}\n".format(self.x_type))
        f.write("y_type: {}\n".format(self.y_type))
        f.write("input_size: {}\n".format(self.input_size))
        f.write("output_size: {}\n".format(self.output_size))
        f.write("learning_rate: {}\n".format(self.learning_rate))
        f.write("weight_init: {}\n".format(self.weight_init))
        f.write("training_file: {}".format(self.training_set.world_state_filename))
        f.close()