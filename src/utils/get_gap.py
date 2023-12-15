import os
import random
import math
import csv
from collections import defaultdict


def calculate_relative_gap(dyn_values, bias_values):
    gaps = []
    for dyn, bias in zip(dyn_values, bias_values):
        if bias != 0:
            gap = ((dyn - bias) / bias) * 100
            gaps.append(gap)
        else:
            gap = 9999999
            gaps.append(gap)
    return sum(gaps) / len(gaps)


def get_gap():
    dyn_dict = defaultdict(list)
    stat_dict = defaultdict(list)

    with open("../../output/output.txt", 'r') as f:
        reader = csv.reader(f, delimiter=';')
        _ = next(reader)

        for row in reader:
            instance, seed, _, _, algorithm, _, betas, _, _, _, _, _, _, _, _, _, _, _, mean_cost_sol,_,_,_,_,_ = row
            # instance seed max_iter_dynamic    max_time    algorithm   selected_random_node    beta    max_iter_random alpha   percentage  n_type_nodes    max_dist    max_vehicles    time    mean_of mean_dynamic_of std_dynamic_of max_dynamic_of min_dynamic_of
            key = (instance, betas)

            if algorithm == 'CONSTRUCTIVE_DYNAMIC':
                dyn_dict[key].append(float(mean_cost_sol))
            elif algorithm == 'CONSTRUCTIVE_STATIC':
                stat_dict[key].append(float(mean_cost_sol))
    print(dyn_dict)
    common_keys = set(dyn_dict.keys()).intersection(set(stat_dict.keys()))

    with open('../../output/gaps.txt', 'w') as f:
        f.write("nombre del problema\tLow/Medium/High\t%relativo\n")

        for key in common_keys:
            dyn_values = dyn_dict[key]
            bias_values = stat_dict[key]
            gap = calculate_relative_gap(dyn_values, bias_values)
            instance, betas = key
            f.write(f"{instance}\t{betas}\t{gap:.3f}%\n")


def build_full_path(relative_path):
    current_path = os.getcwd()

    # Encuentra la posición de "Dinamic_CDP" en la ruta actual
    index = current_path.find(Keys.DYNAMIC_CDP)

    # Si no encontramos la raíz del proyecto, lanzar un error
    if index == -1:
        raise ValueError("The root" + Keys.DYNAMIC_CDP + "is not found in the current path.")

    # Construir la ruta completa hasta "Dinamic_CDP"
    base_path = current_path[:index + len(Keys.DYNAMIC_CDP)]

    # Retorna la ruta completa
    return os.path.join(base_path, relative_path)


def get_random_position(size, beta):
    index = int(math.log(random.random()) / math.log(1 - beta))
    index = index % size
    return index


def read_json():
    import json

    # Read JSON data from a file
    with open("CDP/src/output/data2.json", "r") as f:
        data = json.load(f)


if __name__ == "__main__":
    get_gap()
