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

    num_instances_per_type = 100   # how many sequences for each colored shape
    num_events_per_scene = 4  # num of events per sequence

    ############################
    # custom shapes/bounds = set True to use
    use_custom = False
    custom_shape_list = ['monomino', 'domino', 'tromino1', 'tetromino1']
    custom_color_list = color_list  # ['black']
    custom_bounds = [0, 4, 0, 4]  # xmin, xmax, ymin,ymax
    #############################

    if use_custom:
        the_world = world.World(custom_shape_list, custom_color_list, world_rows, world_columns, custom_bounds)
        the_world.generate_world(0, num_instances_per_type, num_events_per_scene)
    else:
        the_world = world.World(shape_list, color_list, world_rows, world_columns, None)
        the_world.generate_world(num_types, num_instances_per_type, num_events_per_scene)

main()
