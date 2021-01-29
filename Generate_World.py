from polyomino_world.world import world


def main():

    shape_list = ['monomino', 'domino', 'tromino1', 'tromino2',
                  'tetromino1', 'tetromino2', 'tetromino3', 'tetromino4', 'tetromino5']
    color_list = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'cyan', 'magenta']

    all_variant_list = [[0],[0,1],[0,1],[0,1,2,3],[0],[0,1],[0,1,2,3],[0,1,2,3,4,5,6,7],[0,1,2,3]]
    variant_list1 = [[0],[0],[0],[0,1],[0],[0],[0,1],[0,1,2,3],[0,1]]
    variant_list2 = [[0],[1],[1],[2,3],[0],[1],[2,3],[4,5,6,7],[2,3]]

    # size of the world. must be >= smallest shape size
    world_rows = 8
    world_columns = 8

    # number of types to generate.
    # if a positive number, will randomly select a shape and color that many times
    # if 0, will do all combinations of shapes and colors
    num_types = 0

    num_instances_per_type = 10   # how many sequences for each colored shape
    num_events_per_scene = 0  # num of events per sequence

    ############################
    # custom shapes/bounds = set True to use
    use_custom = True#False
    custom_shape_list = shape_list
    custom_color_list = color_list
    current_variant_list = all_variant_list
    custom_bounds = [0,8,0,8] # [0, 8, 0, 8]  # xmin, xmax, ymin,ymax
    background_color = 'grey' #'random'

    name = "full_100"
    #############################
    # adn this
    if use_custom:
        the_world = world.World(custom_shape_list, custom_color_list,
                                world_rows, world_columns, custom_bounds,
                                num_types, num_instances_per_type, num_events_per_scene, background_color, current_variant_list, name)
        the_world.generate_world()

    else:
        the_world = world.World(shape_list, color_list,
                                world_rows, world_columns, None,
                                num_types, num_instances_per_type, num_events_per_scene, background_color, current_variant_list2, name)
        the_world.generate_world()

main()
