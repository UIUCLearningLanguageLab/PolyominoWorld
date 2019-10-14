from src.world import world


def main():

    shape_list = ['monomino', 'domino', 'tromino1', 'tromino2',
                  'tetromino1', 'tetromino2', 'tetromino3', 'tetromino4', 'tetromino5']
    color_list = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'cyan', 'magenta']

    # size of the world. must be >= smallest shape size
    world_rows = 8
    world_columns = 8

    # number of types to generate.
    # if a positive number, will randomly select a shape and color that many times
    # if 0, will do all combinations of shapes and colors
    num_types = 0

    num_instances_per_type = 1   # how many scenes for each colored shape
    num_events_per_scene = 10  # num of events of

    the_world = world.World(shape_list, color_list, world_rows, world_columns)

    the_world.generate_world(num_types, num_instances_per_type, num_events_per_scene)


main()
