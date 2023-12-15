import time

import numpy as np

from src.utils.read_inputs import read, read_run
from src.utils.Constants import *
from src.utils.draw import *
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
    instance_dict[Key.NODES], instance_dict[Key.MAX_DIST], instance_dict[Key.MAX_VEHICLES] = (
        read(instance_dict[Key.INSTANCE]))
    beta0, beta1, beta2, beta3 = getattr(Betas, instance_dict["beta"])
    instance_dict[Key.BLACKBOX] = BlackBox()
    instance_dict[Key.BLACKBOX].setter_betas(beta0, beta1, beta2, beta3)
    instance_dict[Key.NEIGHBOUR_LIMIT] = neighbour(instance_dict[Key.NODES], instance_dict[Key.PERCENTAGE])
    instance_dict[Key.DICT_OF_TYPE] = {k.id: k.id % instance_dict[Key.N_TYPE_NODES] for k in instance_dict[Key.NODES]}
    list_arguments = [Key.NODES, Key.MAX_DIST, Key.SEED, Key.MAX_VEHICLES, Key.ALPHA, Key.NEIGHBOUR_LIMIT, Key.BLACKBOX,
                      Key.DICT_OF_TYPE, Key.MAX_ITER_DYNAMIC, Key.MAX_ITER_RANDOM, Key.BETA_BIAS]
    filtered_args = {key: instance_dict[key] for key in instance_dict if key in list_arguments}

    return Solution(**filtered_args), instance_dict


def save_reduced_information_to_txt(info, mean_of, mean_dynamic_of, std_dynamic_of, max_dynamic_of,
                                    min_dynamic_of, time, min_dynamic_of_100, min_dynamic_of_10,
                                    filename="output\\output.txt"):
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
    reduced_info['std_dynamic_of'] = std_dynamic_of
    reduced_info['max_dynamic_of'] = max_dynamic_of
    reduced_info['min_dynamic_of'] = min_dynamic_of
    reduced_info['min_dynamic_of_100'] = min_dynamic_of_100
    reduced_info['min_dynamic_of_10'] = min_dynamic_of_10


    values_to_save = [v for k, v in reduced_info.items()]

    # Creating a new file or appending to an existing file
    with open(filename, "a") as file:
        file.write(";".join(map(str, values_to_save)) + "\n")


def calculate_delta(instance_dict):
    best_delta = 0
    best_beta = 0
    best_value = 0
    max_iteration_random = instance_dict["max_iter_random"]
    instance_dict["max_iter_random"] = 50
    max_iter_dynamic = instance_dict["max_iter_dynamic"]
    instance_dict["max_iter_dynamic"] = 1

    for delta in range(6, 10):
        for beta in range(6, 10):
            instance_dict["delta"] = delta/10
            instance_dict["beta_bias"] = beta/10

            list_arguments = [Key.NODES, Key.MAX_DIST, Key.SEED, Key.MAX_VEHICLES, Key.ALPHA, Key.NEIGHBOUR_LIMIT,
                              Key.BLACKBOX,
                              Key.DICT_OF_TYPE, Key.MAX_ITER_DYNAMIC, Key.MAX_ITER_RANDOM, Key.BETA_BIAS]
            filtered_args = {key: instance_dict[key] for key in instance_dict if key in list_arguments}
            solution = Solution(**filtered_args)
            result = solution.run(Algorithm.CONSTRUCTIVE_STATIC, Algorithm.SELECT_SAVING_GRASP, instance_dict)
            if best_value < np.mean(result[1]):
                best_value = np.mean(result[1])
                best_delta = delta/10
                best_beta = beta/10

    instance_dict["max_iter_dynamic"] = max_iter_dynamic
    instance_dict["max_iter_random"] = max_iteration_random
    instance_dict["delta"] = best_delta
    instance_dict["beta_bias"] = best_beta

    return instance_dict


if __name__ == '__main__':
    instance = read_run()
    for instance_dict in instance:
        solution, instance_dict = create_instance(instance_dict)
        algo = instance_dict[Key.ALGORITHM]
        selected_procedure = instance_dict[Key.SELECTED_NODE_FUNCTION]
        instance_dict = calculate_delta(instance_dict)
        start = time.time()
        print(instance_dict[Key.BETA_BIAS])
        print(instance_dict[Key.ALPHA])
        list_arguments = [Key.NODES, Key.MAX_DIST, Key.SEED, Key.MAX_VEHICLES, Key.ALPHA, Key.NEIGHBOUR_LIMIT,
                          Key.BLACKBOX,
                          Key.DICT_OF_TYPE, Key.MAX_ITER_DYNAMIC, Key.MAX_ITER_RANDOM, Key.BETA_BIAS]
        filtered_args = {key: instance_dict[key] for key in instance_dict if key in list_arguments}
        solution = Solution(**filtered_args)

        results = solution.run(algo, selected_procedure, instance_dict)
        save_reduced_information_to_txt(instance_dict, np.mean(results[1]), np.mean(results[2]), np.std(results[2]),
                                        np.min(results[2]), np.max(results[2]), time.time()-start,
                                        np.min(results[2][:100]), np.min(results[2][:10]))

        # maps = Map(instance_dict["nodes"])
        # maps.print_route(results[0][-1])

        print(instance_dict[Key.ALGORITHM])
        print("Mean OF: " + str(np.mean(results[1])))
        print("Mean Dynamic OF: " + str(np.mean(results[2])))
        print()
