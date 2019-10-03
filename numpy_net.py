import sys
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from src import config
from src import world
import heatmapcluster


class NeuralNetwork:
    ############################################################################################################
    def __init__(self,dataset,input_size, hidden_size, output_size, learning_rate, weight_init,batch_size):

        self.dataset=dataset
        self.batch_size=batch_size
        self.batched_x,self.batched_y = self.batch_data()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight_mean = weight_init[0]
        self.weight_stdev = weight_init[1]

        self.h_bias = np.random.normal(self.weight_mean, self.weight_stdev, [self.hidden_size])
        self.h_x = np.random.normal(self.weight_mean, self.weight_stdev, [self.hidden_size, self.input_size])

        self.o_bias = np.random.normal(self.weight_mean, self.weight_stdev, [self.output_size])
        self.o_h = np.random.normal(self.weight_mean, self.weight_stdev, [self.output_size, self.hidden_size])

        self.learning_rate = learning_rate

        np.set_printoptions(suppress=True, precision=3,floatmode='fixed',linewidth=np.inf)
    ############################################################################################################
    def feedforward(self, x):
        h = self.tanh(np.dot(self.h_x, x) + self.h_bias)
        o = self.sigmoid(np.dot(self.o_h, h) + self.o_bias)
        return h, o

    ############################################################################################################
    @staticmethod
    def calc_cost(y, o):
        return y - o
    ############################################################################################################
    def backpropogation(self, x, o, h, o_cost):
        o_delta = o_cost * self.sigmoid_prime(o)

        h_cost = np.dot(o_delta, self.o_h)
        h_delta = h_cost * self.tanh_prime(h)

        self.o_bias -= o_delta * self.learning_rate
        self.o_h += (np.dot(o_delta.reshape(len(o_delta), 1), h.reshape(1, len(h))) * self.learning_rate)

        self.h_bias -= h_delta * self.learning_rate
        self.h_x += (np.dot(h_delta.reshape(len(h_delta), 1), x.reshape(1, len(x))) * self.learning_rate)

    ############################################################################################################
    @staticmethod
    def tanh(z):
        return np.tanh(z)

    ############################################################################################################
    @staticmethod
    def tanh_prime(z):
        return 1.0 - np.tanh(z)**2

    ############################################################################################################
    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    ############################################################################################################
    @staticmethod
    def sigmoid_prime(z):
        return 1/(1+np.exp(-z)) * (1 - 1/(1+np.exp(-z)))
    ###########################################################################################################
    def batch_data(self):
        data=self.dataset
        ##get list of x and list of y
        np_x = []
        np_y = []
        x_batches=[]
        y_batches=[]
        for i in range(len(data)):
            np_x.append(np.array(data[i].image_matrix.flatten()))
            np_y.append(np.array(data[i].label_matrix.flatten()))

        x_batches=[np_x[i*self.batch_size:(i+1)*self.batch_size]for i in range((len(np_x) + self.batch_size - 1) // self.batch_size )]
        y_batches=[np_y[i*self.batch_size:(i+1)*self.batch_size]for i in range((len(np_y) + self.batch_size - 1) // self.batch_size )]

        return x_batches, y_batches

    ###########################################################################################################
    def train(self,num_epochs):
        for i in range(num_epochs):
            epoch_cost = 0
            for k in range(len(self.batched_x)):
                x_batch = self.batched_x[k]
                y_batch = self.batched_y[k]
                for j in range(len(x_batch)):
                    x = np.array(x_batch[j], float)
                    y = np.array(y_batch[j], float)
                    h, o = self.feedforward(x)
                    o_cost = self.calc_cost(y, o)
                    epoch_cost += (o_cost ** 2).sum()
                    self.backpropogation(x, o, h, o_cost)
            if i % 10 == 0:
                if config.PrintOptions.print_items:
                    print("{} Cost: {} ".format(i, epoch_cost))

            if i % 999 == 0:
                for k in range(len(self.batched_x)):
                    x_batch = self.batched_x[k]
                    y_batch = self.batched_y[k]
                    for i in range(len(x_batch)):
                        x = np.array(x_batch[i], float)
                        y = np.array(y_batch[i], float)
                        h, o = self.feedforward(x)
                        o_cost = self.calc_cost(y, o)
                        prob_list=self.get_prob_outputs(o)
                        shape_ans,size_ans,color_ans=self.get_correct_out(y)
                        max_shape_index,max_size_index,max_color_index=self.get_max_indices(prob_list)

                        if shape_ans=='Monomino' and color_ans=='red':
                            print("_______________________________________")
                            config.PrintOptions.print_green("       "+shape_ans + " " + str(size_ans)+ " "+ color_ans)
                            self.print_prob_outputs(prob_list,y,max_shape_index,max_size_index,max_color_index)
                            print("_______________________________________")
                        #print("x:", x.sum(), "\ny:", y, "\no:", o, "\ncost:", '{:0.3f}'.format((o_cost ** 2).sum()))
                print("\n")
    ###########################################################################################################
    def test(self,test_data):
        x_test=[]
        y_test=[]
        for i in range(len(test_data)):
            x_test.append(np.array(test_data[i].image_matrix.flatten()))
            y_test.append(np.array(test_data[i].label_matrix.flatten()))
        for i in range(len(x_test)):
            x=np.array(x_test[i],float)
            y=np.array(y_test[i],float)
            h,o = self.feedforward(x)
            o_cost = self.calc_cost(y, o)
            prob_list=self.get_prob_outputs(o)
            shape_ans,size_ans,color_ans=self.get_correct_out(y)
            max_shape_index,max_size_index,max_color_index=self.get_max_indices(prob_list)
            print("_______________________________________")
            config.PrintOptions.print_green("       "+shape_ans + " " + str(size_ans)+ " "+ color_ans)
            self.print_prob_outputs(prob_list,y,max_shape_index,max_size_index,max_color_index)
            print("_______________________________________")

        print("\n")
    ###########################################################################################################
    def plot_heatmap(self):
        h1 = heatmapcluster.heatmapcluster(self.h_x.transpose(),
                                            cmap=plt.cm.coolwarm,
                                            show_colorbar=True, colorbar_pad=2,
                                             top_dendrogram=True)
        plt.show()
    ################################################################################################################################
    def save_network(self):
        file = open('np_network.txt', 'wb')
        pickle.dump(self,file)
    ################################################################################################################################
    def load_network(self):
        file=open('np_network.txt','rb')
        dataset= pickle.load(file)
        return dataset
    ################################################################################################################################
    def get_prob_outputs(self,o):
        prob_list = np.zeros(21) #num_features
        shape_total=0
        size_total=0
        color_total=0
        for i in range(9):
            shape_total+=o[i];
        for i in range(9,13):
            size_total+=o[i];
        for i in range(13,21):
            color_total+=o[i];
        for i in range(9):
            prob_list[i]=o[i]/shape_total;
        for i in range(9,13):
            prob_list[i]=o[i]/size_total;
        for i in range(13,21):
            prob_list[i]=o[i]/color_total;
        return prob_list;
    ################################################################################################################################
    def print_prob_outputs(self,prob_list,y,max_shape_index,max_size_index,max_color_index):
        percents = [str(round(i*100,3)) for i in prob_list]
        shapes = config.Shape.shape_list
        sizes = config.Shape.shape_sizes
        colors = config.Shape.shape_color_list
        #print("{} {} {}".format(len(shapes),len(sizes),len(colors)))

        for i in range(len(shapes)):
            if i == max_shape_index:
                if y[i]==1:
                    config.PrintOptions.print_green(shapes[i] +"   "+ percents[i]+" %")
                else:
                    config.PrintOptions.print_red(shapes[i] +"   "+ percents[i]+" %")
            else:
                print(shapes[i] +"   "+ percents[i]+" %")

        for i in range(len(sizes)):
            if i+9 == max_size_index:
                if y[i+9]==1:
                    config.PrintOptions.print_green(str(sizes[i]) +"   "+ percents[i+9]+" %")
                else:
                    config.PrintOptions.print_red(str(sizes[i]) +"   "+ percents[i+9]+" %")
            else:
                print(str(sizes[i]) +"    "+percents[i+9]+" %")

        for i in range(len(colors)):
            if i+13 == max_color_index:
                if y[i+13]==1:
                    config.PrintOptions.print_green(str(colors[i]) +"   "+ percents[i+13]+" %")
                else:
                    config.PrintOptions.print_red(str(colors[i]) +"   "+ percents[i+13]+" %")
            else:
                print(str(colors[i]) +"    "+percents[i+13]+" %")
    ################################################################################################################################
    def get_max_indices(self,prob_list):
        max_shape_index=0
        max_size_index=9
        max_color_index=13
        for i in range(9):
            if prob_list[i] > prob_list[max_shape_index]:
                max_shape_index=i;
        for i in range(9,13):
            if prob_list[i] > prob_list[max_size_index]:
                max_size_index=i;
        for i in range(13,21):
            if prob_list[i] > prob_list[max_color_index]:
                max_color_index=i;
        return max_shape_index,max_size_index,max_color_index
    ################################################################################################################################
    def get_correct_out(self,y):
        for i in range(9):
            if y[i]==1:
                shape_ans=config.Shape.shape_list[i]
        for i in range(9,13):
            if y[i]==1:
                size_ans=config.Shape.shape_sizes[i-9]
        for i in range(13,21):
            if y[i]==1:
                color_ans=config.Shape.shape_color_list[i-13]
        return shape_ans,size_ans,color_ans
