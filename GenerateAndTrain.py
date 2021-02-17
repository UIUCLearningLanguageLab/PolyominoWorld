import random
import numpy as np
from polyomino_world.world import world
from polyomino_world.networks import dataset, network, analysis

def main():
	################### Generate ######################
    # size of the world. must be >= smallest shape size
    world_rows = 8
    world_columns = 8
    custom_bounds = [0, 8, 0, 8]  # [0, 8, 0, 8]  # xmin, xmax, ymin, ymax
    background_color = 'grey'  # 'random'

    shape_list = ['monomino', 'domino', 'tromino1', 'tromino2',
                  'tetromino1', 'tetromino2', 'tetromino3', 'tetromino4', 'tetromino5']
    color_list = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'cyan', 'magenta']

    variant_list = [[0],
                    [0, 1],
                    [0, 1], [0, 1, 2, 3],
                    [0], [0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3]]
    variant_list_a = [[0],
                      [0],
                      [0], [0, 1],
                      [0], [0], [0, 1], [0, 1, 2, 3], [0, 1]]
    variant_list_b = [[0],
                      [1],
                      [1], [2, 3],
                      [0], [1], [2, 3], [4, 5, 6, 7], [2, 3]]

    # number of types to generate. if a positive number, will randomly select a shape and color that many times
    # if 0, will do all combinations of shapes and colors, num_instances_per_type times, in random locations
    # if -1, will do exactly 1 of all combinations of shapes and colors, in all possible locations
    location_type = "all"
    num_instances_per_type = 10   # how many sequences for each colored shape
    num_events_per_scene = 0  # num of events per sequence

    name = ""
    train_random_seed_list = []
    random_range = len(shape_list)*len(color_list)*num_instances_per_type*3
    for i in range(random_range):
    	train_random_seed_list.append(random.randrange(random_range))

    train_world = world.World(shape_list, color_list,
                            world_rows, world_columns, custom_bounds,
                            location_type, num_instances_per_type, num_events_per_scene,
                            background_color, variant_list, name, train_random_seed_list)
    train_world.generate_world()


    test_random_seed_list = []
    random_range = len(shape_list)*len(color_list)*num_instances_per_type*3
    for i in range(random_range):
    	test_random_seed_list.append(random.randrange(random_range))

    test_world = world.World(shape_list, color_list,
                            world_rows, world_columns, custom_bounds,
                            location_type, num_instances_per_type, num_events_per_scene,
                            background_color, variant_list, name, test_random_seed_list)
    test_world.generate_world()

    traning_info_dict = {"size": "{}x{}".format(world_rows, world_columns),
    			 "colors": "red-blue-green-magenta-yellow-cyan-white-black",
    			 "shapes": "monomino-domino-tromino1-tromino2-tetromino1-tetromino2-tetromino3-tetromino4-tetromino5",
    			 "variants": "[0][0-1][0-1-2-3][0-1-2-3-4-5-6-7][0-1-2-3]",
    			 "background": background_color,
    			 "num_events": num_events_per_scene,
    			 "num_instances": num_instances_per_type,
    			 "location_type": location_type,
    			 "random_seeds": str(train_random_seed_list)}

    ################### Train ######################

    np.set_printoptions(precision=4, suppress=True)

    hidden_size = 16
    hidden_activation_function = 'tanh'
    learning_rate = 0.3
    num_epochs = 10
    weight_init = 0.00001
    output_freq = 5 #25
    verbose = False
    x_type = 'WorldState'
    y_type = 'FeatureVector'  # 'WorldState'
    included_features = [1, 1, 1, 0]  # Include: Shape, Size, Color, Action
    shuffle_sequences = True
    shuffle_events = False
    processor = 'CPU'
    optimizer = 'SGD'

    # training_file = "w8-8_s9_c8_location-all_10_0_allpossible_8x8_topbottom.csv"
    # test_file = "w8-8_s9_c8_location-all_10_0_allpossible_8x8_topbottom.csv"
    training_dataset = train_world
    test_dataset = test_world

    training_set = dataset.DataSet(training_dataset, None, included_features, processor)
    test_set = dataset.DataSet(test_dataset, None, included_features, processor)

    net = network.MlNet()

    net.init_model(x_type, y_type, training_set,
                   hidden_size, hidden_activation_function, optimizer, learning_rate, weight_init, processor, traning_info_dict)
    # net.load_model(network_directory, included_features, processor)

    analysis.train_a(net, training_set, test_set, num_epochs, optimizer, learning_rate,
                     shuffle_sequences, shuffle_events, output_freq, verbose)


main()
