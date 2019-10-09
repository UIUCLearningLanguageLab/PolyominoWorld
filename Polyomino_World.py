from src import world
from src import config


def generate_data(the_world, num_scenes, scene_length, file_name):
    shape_counter = 0
    scene_counter = 0
    f = open(file_name, 'w')
    f.close()

    for i in range(9):  # num of shape types/size
        for j in range(8):  # num of colors
            for k in range(num_scenes):
                the_world.init_world(scene_counter)
                shape_name = config.Shape.shape_list[i]
                shape_color = config.Shape.color_list[j]
                the_world.add_shape_to_world(shape_name, shape_counter, shape_color)
                the_world.save_world_state(file_name)
                shape_counter += 1

                for m in range(scene_length):
                    the_world.next_turn()
                    the_world.save_world_state(file_name)
                scene_counter += 1
    n = (i+1)*(j+1)
    print("Created dataset {} with {} objects ({}*{}), containing {} events each".format(file_name, scene_counter,
                                                                                         n, num_scenes, scene_length))


def main():

    train_n = 1   # how many scenes for each colored shape
    test_n = 1

    event_n = 10  # num of events of

    the_world = world.World()

    generate_data(the_world, train_n, event_n, "training_all_1_10.csv")
    generate_data(the_world, test_n, event_n, "test_all_1_10.csv")


main()
