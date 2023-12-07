import time

import numpy as np

from src.utils.read_inputs import read, read_run
from src.utils.Constants import *
from src.utils.maps import *
from src.utils.classes import *
from src.algorithm.solution import Solution


def neighbour(nodes, percentage):
    maxim = 0
    for i in range(len(nodes) - 2):
        for j in range(len(nodes) - 2):
            if i == j:
                continue
            edge_a_b = Edge(nodes[i + 1], nodes[j + 1])
            if edge_a_b.distance > maxim:
                maxim = edge_a_b.distance

    return maxim*percentage


def create_instance(instance_dict):
    instance_dict[Key.NODES], instance_dict[Key.MAX_DIST], instance_dict[Key.MAX_VEHICLES] = read(instance_dict[Key.INSTANCE])
    beta0, beta1, beta2, beta3 = Betas().MEDIUMTEST
    instance_dict[Key.BLACKBOX] = BlackBox()
    instance_dict[Key.BLACKBOX].setter_betas(beta0, beta1, beta2, beta3)
    instance_dict[Key.NEIGHBOUR_LIMIT] = neighbour(instance_dict[Key.NODES], instance_dict[Key.PERCENTAGE])
    instance_dict[Key.DICT_OF_TYPE] = {k.id: k.id % instance_dict[Key.N_TYPE_NODES] for k in instance_dict[Key.NODES]}
    list_arguments = [Key.NODES, Key.MAX_DIST, Key.SEED, Key.MAX_VEHICLES, Key.ALPHA, Key.NEIGHBOUR_LIMIT, Key.BLACKBOX,
                      Key.DICT_OF_TYPE, Key.MAX_ITER_DYNAMIC, Key.MAX_ITER_RANDOM]
    filtered_args = {key: instance_dict[key] for key in instance_dict if key in list_arguments}

    return Solution(**filtered_args), instance_dict


def save_reduced_information_to_txt(info, mean_of, mean_dynamic_of, time, filename="..\\output\\output.txt"):
    """
    Saves reduced information along with additional mean values to a text file, excluding certain keys.

    :param info: Dictionary containing the main information to be saved.
    :param mean_of: Mean value to be saved under the key 'mean_of'.
    :param mean_dynamic_of: Mean dynamic value to be saved under the key 'mean_dynamic_of'.
    :param filename: Name of the file to save the information to. Defaults to 'output.txt'.
    """
    # Keys to be excluded from saving
    keys_to_exclude = ['bb', 'nodes', 'dict_of_types', 'neighbour_limit']

    # Removing the specified keys from the dictionary
    reduced_info = {k: v for k, v in info.items() if k not in keys_to_exclude}

    # Adding mean values to the reduced information dictionary
    reduced_info['time'] = time
    reduced_info['mean_of'] = mean_of
    reduced_info['mean_dynamic_of'] = mean_dynamic_of

    values_to_save = [v for k, v in reduced_info.items()]

    # Creating a new file or appending to an existing file
    with open(filename, "a") as file:
        file.write(";".join(map(str, values_to_save)) + "\n")


if __name__ == '__main__':
    instance = read_run()
    for instance_dict in instance:
        solution, instance_dict = create_instance(instance_dict)
        # print(instance_dict[Key.ALGORITHM])
        algo = instance_dict[Key.ALGORITHM]
        selected_procedure = instance_dict[Key.SELECTED_NODE_FUNCTION]
        start = time.time()
        results = solution.run(algo, selected_procedure, instance_dict)
        save_reduced_information_to_txt(instance_dict, np.mean(results[1]), np.mean(results[2]), time.time()-start)

        print(instance_dict[Key.ALGORITHM])
        print("Mean OF: " + str(np.mean(results[1])))
        print("Mean Dynamic OF: " + str(np.mean(results[2])))
        print()
