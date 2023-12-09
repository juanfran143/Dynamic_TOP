
class Betas:
    LOW = [{0: 0.7, 1: 0.8, 2: 0.9, 3: 1, 4: 1.1},
              [-90000, -60000, -10000, 0, 2],
              {0: -0.3, 1: -0.2, 2: -0.15, 3: -0.1, 4: -0.05},
              {0: -1.5, 1: -1, 2: -0.5, 3: -0.25, 4: -0.1}]

    MEDIUM = [{0: 0.7, 1: 0.8, 2: 0.9, 3: 1, 4: 1.1},
              [-0.8, -0.55, 0, 0.4, 0.6],
              {0: -0.3, 1: -0.2, 2: -0.15, 3: -0.1, 4: -0.05},
              {0: -2.5, 1: -2, 2: -0.5, 3: -0.25, 4: -0.1}]

    MEDIUMTEST = [{0: 0, 1: 0.0, 2: 0.0, 3: 0, 4: 0},
              [-0.8, -0.55, 0, 0.4, 0.6],
              {0: -0.3, 1: -0.2, 2: -0.15, 3: -0.1, 4: -0.05},
              {0: 2.5, 1: 2, 2: -0.5, 3: -0.25, 4: -0.1}]

    HIGHTEST = [
        {0: 0, 1: 0.0, 2: 0.0, 3: 0, 4: 0},
        [-8.9, -7.3, -6.9, -0.2, -0.05],
        {0: -8.2, 1: -7.8, 2: -5.5, 3: -0.6, 4: -0.3},
        {0: -8.5, 1: -7.5, 2: -5.2, 3: -0.5, 4: -0.2}
    ]

    HIGH = [{0: 0.5, 1: 0.6, 2: 1, 3: 1.2, 4: 1.3},
              [-1.85, -0.8, 0, 0.6, 0.8],
              {0: -0.6, 1: -0.4, 2: -0.25, 3: -0.15, 4: -0.1},
              {0: -3, 1: -2.5, 2: -0.75, 3: -0.45, 4: -0.2}]

    VERYHIGH = [{0: -0.4, 1: -0.2, 2: 0.75, 3: 1, 4: 1.1},
              [-2, -1.2, -1, 0.3, 0.5],
              {0: -1, 1: -0.6, 2: -0.5, 3: -0.25, 4: -0.1},
              {0: -3.2, 1: -2.7, 2: -1.75, 3: -0.45, 4: -0.2}]

    EXTREME = [{0: -100, 1: -500, 2: -100, 3: -100, 4: -100},
              [-0.8, -0.55, 0, 0.4, 0.6],
              {0: -0.3, 1: -0.2, 2: -0.15, 3: -0.1, 4: -0.05},
              {0: -2.5, 1: -2, 2: -0.5, 3: -0.25, 4: -0.1}]


class Algorithm:
    STATIC = "STATIC"
    DYNAMIC = "DYNAMIC"
    CONSTRUCTIVE_DYNAMIC = "CONSTRUCTIVE_DYNAMIC"
    CONSTRUCTIVE_STATIC = "CONSTRUCTIVE_STATIC"
    SELECT_SAVING_GREEDY = "GREEDY"
    SELECT_SAVING_GRASP = "GRASP"


class Key:
    ROOT_FOLDER = "Dynamic_TOP"
    INSTANCE_FOLDER = "Instances"
    FOLDER_RUN = "run"

    NODES = "nodes"
    MAX_DIST = "max_dist"
    MAX_VEHICLES = "max_vehicles"
    BLACKBOX = "bb"
    NEIGHBOUR_LIMIT = "neighbour_limit"
    DICT_OF_TYPE = "dict_of_types"
    N_TYPE_NODES = "n_type_nodes"
    BETA_BIAS = "beta_bias"
    PERCENTAGE = "percentage"
    SEED = "seed"
    ALPHA = "alpha"
    MAX_ITER_RANDOM = "max_iter_random"
    MAX_ITER_DYNAMIC = "max_iter_dynamic"
    MAX_TIME = "max_time"
    INSTANCE = "instance"
    BETA_BLACKBOX = "beta"

    ALGORITHM = "algorithm"
    SELECTED_NODE_FUNCTION = "selected_random_node"