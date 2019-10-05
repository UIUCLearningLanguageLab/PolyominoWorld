from src import world
from src import config


def generate_data(the_world, num_events, event_length, file_name):
    shape_counter = 0
    event_counter = 0
    f = open(file_name, 'w')
    f.close()

    for i in range(9):  # num of shape types/size
        for j in range(8):  # num of colors
            print(file_name, config.Shape.shape_list[i], config.Shape.color_list[j])
            for k in range(num_events):
                the_world.init_world(event_counter)
                shape_name = config.Shape.shape_list[i]
                shape_color = config.Shape.color_list[j]
                the_world.add_shape_to_world(shape_name, shape_counter, shape_color)
                the_world.save_world_state(file_name)
                shape_counter += 1

                for m in range(event_length):
                    the_world.next_turn()
                    the_world.save_world_state(file_name)
                event_counter += 1


def main():

    event_n = 5
    train_n = 5
    test_n = 1
    the_world = world.World()

    generate_data(the_world, train_n, event_n, "training.csv")
    generate_data(the_world, test_n, event_n, "test.csv")


main()
