from src import world
from src import display
from src import shapes
from src import config
import pickle
from PIL import Image
import numpy as np


def generate_data(the_world, num_training_instances, master_shape_list, color_list, num_images, file_name):
    shape_counter = 0
    shape_list = []
    event_counter = 0
    for i in range(9):  # num of shape types/size
        for j in range(8):  # num of colors
            for k in range(num_training_instances):
                the_world.init_world(event_counter)
                current_shape = master_shape_list[i]
                current_shape.init_shape(shape_counter, color_list[j])
                the_world.add_shape_to_world(current_shape, shape_counter)
                shape_counter += 1
                # sequence_list = []
                the_world.save_world_state(file_name)
                print(shape_counter, current_shape.size, current_shape.name, current_shape.color, k+1)

                for m in range(num_images):
                    the_world.next_turn()
                    the_world.save_world_state(file_name)
                    # the_display.next()
                    # the_display.canvas.postscript(file="canvas.eps", colormode='color',
                    #                               width=config.World.num_columns, height=config.World.num_rows,
                    #                               pagewidth=config.World.num_columns-1,
                    #                               pageheight=config.World.num_rows-1)
                    # img = Image.open("canvas.eps")
                    # img = np.array(img)
                    # label_vector = np.zeros([the_world.num_features])
                    # label_vector[the_world.feature_index_dict[current_shape.size]] = 1
                    # label_vector[the_world.feature_index_dict[current_shape.color]] = 1
                    # label_vector[the_world.feature_index_dict[current_shape.name]] = 1
                    # img_data = config.Data(current_shape.id_number, current_shape.name,
                    #                        current_shape.size, current_shape.color, img, label_vector)
                #     sequence_list.append(img_data)
                # shape_list.append(sequence_list)
                event_counter += 1

    return shape_list


def main():

    num_images = 5
    num_training_instances = 1
    num_test_instances = 0
    the_world = world.World()

    shape_list = [shapes.Monomino(the_world),
                  shapes.Domino(the_world),
                  shapes.Tromino1(the_world),
                  shapes.Tromino2(the_world),
                  shapes.Tetromino1(the_world),
                  shapes.Tetromino2(the_world),
                  shapes.Tetromino3(the_world),
                  shapes.Tetromino4(the_world),
                  shapes.Tetromino5(the_world)]

    color_list = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'cyan', 'magenta']

    #the_display = display.Display(the_world)

    train_data_list = generate_data(the_world, num_training_instances, shape_list, color_list, num_images, "training.txt")
    test_data_list = generate_data(the_world, num_test_instances, shape_list, color_list, num_images, "test.txt")

    # write Data
    print("training data list size = {}".format(len(train_data_list)))
    file = open('dataset.txt', 'wb')
    pickle.dump(train_data_list, file)

    print("testing data list size = {}".format(len(test_data_list)))
    f = open('test_dataset.txt', 'wb')
    pickle.dump(test_data_list, f)

    #the_display.root.mainloop()


main()
