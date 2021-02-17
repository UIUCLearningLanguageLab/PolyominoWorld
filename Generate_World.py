from polyomino_world.world import world


def main():
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

    the_world = world.World(shape_list, color_list,
                            world_rows, world_columns, custom_bounds,
                            location_type, num_instances_per_type, num_events_per_scene,
                            background_color, variant_list, name)
    the_world.generate_world()
    the_world.print_world_history()


main()
