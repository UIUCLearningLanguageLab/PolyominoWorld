import numpy_net
import torch_net
import numpy as np
import pickle
import matplotlib as plt

def main():

#****** Run Polyomino_World.py to generate data*************
    file=open('dataset.txt','rb')
    dataset= pickle.load(file)

    f=open('test_dataset.txt','rb')
    test_data=pickle.load(f)

    input_size = 300
    hidden_size = 64
    output_size = 21
    learning_rate = 0.3
    num_epochs = 1000
    weight_init = [0, 0.0000001]
    batch_size = 3600

###############################################################################
#                            uncomment network to run
###############################################################################
    # net = torch_net.PytorchNet(dataset,input_size,
    #                        hidden_size, output_size,
    #                        learning_rate, weight_init,
    #                        batch_size)
    #
    net = numpy_net.NeuralNetwork(dataset,input_size,
                                    hidden_size,output_size,
                                    learning_rate,weight_init,
                                    batch_size)
################################################################################
#                            Train or test

    net.train(num_epochs)
    net.save_network()
    #network = net.load_network()
    #network.test(test_data)
    #net.plot_heatmap()
    # print("length test {}".format(len(test_data)))
    # print("length dataset{}".format(len(dataset)))

main()
