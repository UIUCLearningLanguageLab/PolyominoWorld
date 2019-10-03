import sys
import torch
from torch import nn
import torch.distributions as tdist
import numpy as np
import seaborn as sns;
import matplotlib.pyplot as plt
import pickle
from src import config

class PytorchNet(nn.Module):
    def __init__(self,dataset,input_size, hidden_size, output_size, learning_rate, weight_init, batch_size):
        super().__init__()

        self.dataset=dataset
        self.batch_size=batch_size
        self.batched_x,self.batched_y = self.batch_data()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight_mean = weight_init[0]
        self.weight_stdev = weight_init[1]
        self.learning_rate= learning_rate

        self.hidden = nn.Linear(self.input_size,self.hidden_size)
        self.output = nn.Linear(self.hidden_size,self.output_size)

        self.hidden.weight = torch.nn.Parameter(torch.randn(self.hidden_size,self.input_size) * self.weight_stdev + self.weight_mean)
        self.output.weight = torch.nn.Parameter(torch.randn(self.output_size,self.hidden_size) * self.weight_stdev + self.weight_mean)

        self.hidden.bias = torch.nn.Parameter(torch.randn(hidden_size) * self.weight_stdev + self.weight_mean)
        self.output.bias = torch.nn.Parameter(torch.randn(output_size) * self.weight_stdev + self.weight_mean)


        self.sigmoid=nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        self.loss_fn = torch.nn.MSELoss(reduction='mean')

        np.set_printoptions(suppress=True, precision=3,floatmode='fixed',linewidth=np.inf,threshold=np.nan)



    def forward(self,x):
        x=self.hidden(x)
        x=self.sigmoid(x)
        x=self.output(x)
        return x

    def backpropogation(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def batch_data(self):
        data=self.dataset
        x_tensors = []
        y_tensors = []
        x_batches=[]
        y_batches=[]
        for i in range(len(data)):
            x_tensors.append(torch.FloatTensor(data[i].image_matrix.flatten()))
            y_tensors.append(torch.FloatTensor(data[i].label_matrix.flatten()))

        x_batches=[x_tensors[i*self.batch_size:(i+1)*self.batch_size]for i in range((len(x_tensors) + self.batch_size - 1) // self.batch_size )]
        y_batches=[y_tensors[i*self.batch_size:(i+1)*self.batch_size]for i in range((len(y_tensors) + self.batch_size - 1) // self.batch_size )]

        return x_batches, y_batches

    def train(self, num_epochs):
        for i in range(num_epochs):
            epoch_cost=0
            for k in range(len(self.batched_x)):
                batch_x = self.batched_x[k]
                batch_y = self.batched_y[k]
                for j in range(len(batch_x)):
                    x=batch_x[j]
                    y=batch_y[j]

                    output = self.forward(x)
                    loss = self.loss_fn(output,y)
                    self.backpropogation(loss)
                    epoch_cost+=loss
            if i % 10 == 0:
                print("{} cost {}".format(i,loss.item()))

            if i % 100 == 0:
                for k in range(len(self.batched_x)):
                    x_batch = self.batched_x[k]
                    y_batch = self.batched_y[k]
                    for i in range(len(x_batch)):
                        x =x_batch[i]
                        y =y_batch[i]
                        o = self.forward(x)
                        o_cost = self.loss_fn(o,y)
                        y_print = np.array([item.item() for item in y])
                        o_print = np.array([item.item() for item in o])
                        prob_list=self.get_prob_outputs(o)
                        shape_ans,size_ans,color_ans=self.get_correct_out(y)
                        max_shape_index,max_size_index,max_color_index=self.get_max_indices(prob_list)
                        print("_______________________________________")
                        config.PrintOptions.print_green("       "+shape_ans + " " + str(size_ans)+ " "+ color_ans)
                        self.print_prob_outputs(prob_list,y,max_shape_index,max_size_index,max_color_index)
                        print("_______________________________________")
                print("\n")
    ###############################################################################################################################
    def test(self,test_data):
        x_test=[]
        y_test=[]
        for i in range(len(test_data)):
            x_test.append(torch.FloatTensor(test_data[i].image_matrix.flatten()))
            y_test.append(torch.FloatTensor(test_data[i].label_matrix.flatten()))
        for i in range(len(x_test)):
            x=x_test[i]
            y=y_test[i]
            o = self.forward(x)
            o_cost = self.loss_fn(o,y)
            y_print = np.array([item.item() for item in y])
            o_print = np.array([item.item() for item in o])
            prob_list=self.get_prob_outputs(o)
            shape_ans,size_ans,color_ans=self.get_correct_out(y)
            max_shape_index,max_size_index,max_color_index=self.get_max_indices(prob_list)
            print("_______________________________________")
            config.PrintOptions.print_green("       "+shape_ans + " " + str(size_ans)+ " "+ color_ans)
            self.print_prob_outputs(prob_list,y,max_shape_index,max_size_index,max_color_index)
            print("_______________________________________")

        print("\n")
    ################################################################################################################################
    def plot_heatmap(self):
        sns.set()
        data=self.output.weight
        ax = sns.heatmap(data.detach())
        plt.show()
    ################################################################################################################################
    def save_network(self):
        file = open('py_network.txt', 'wb')
        pickle.dump(self,file)
    ################################################################################################################################
    def load_network(self):
        file=open('py_network.txt','rb')
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
