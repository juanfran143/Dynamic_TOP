
class Betas:
    LOW = [
        {0: 0, 1: 0.0, 2: 0.0, 3: 0, 4: 0},
        {0: 0, 1: -0.1, 2: -0.2, 3: -0.3, 4: -0.4},
        {0: -1, 1: -0.8, 2: -0.6, 3: -0.4, 4: -0.2},
        {0: 1, 1: 1.1, 2: 1.2, 3: 1.3, 4: 1.4}
    ]

    MEDIUM = [
        {0: 0, 1: 0.0, 2: 0.0, 3: 0, 4: 0},
        {0: 0, 1: -0.2, 2: -0.4, 3: -0.6, 4: -0.8},
        {0: -1.2, 1: -1, 2: -0.8, 3: -0.6, 4: -0.4},
        {0: 1.2, 1: 1.4, 2: 1.6, 3: 1.8, 4: 2}
    ]

    HIGH = [
        {0: 0, 1: 0.0, 2: 0.0, 3: 0, 4: 0},
        {0: 0, 1: -0.5, 2: -1, 3: -1.5, 4: -2},
        {0: -2, 1: -1.5, 2: -1, 3: -0.8, 4: -0.5},
        {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    ]


class Algorithm:
    STATIC = "STATIC"
    DYNAMIC = "DYNAMIC"
    CONSTRUCTIVE_DYNAMIC = "CONSTRUCTIVE_DYNAMIC"
    CONSTRUCTIVE_STATIC = "CONSTRUCTIVE_STATIC"
    CONSTRUCTIVE_DYNAMIC_BB = "CONSTRUCTIVE_DYNAMIC_BB"
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
    STANDARD = "standard"

    ALGORITHM = "algorithm"
    SELECTED_NODE_FUNCTION = "selected_random_node"