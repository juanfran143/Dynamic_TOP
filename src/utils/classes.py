import math
import random
from itertools import product
import numpy as np


class Node:

    def __init__(self, id_, reward, x, y):
        self.id = id_
        self.reward = reward
        self.x = x
        self.y = y
        self.route = None

    def distance(self, end):
        return ((self.x - end.x) ** 2 + (self.y - end.y) ** 2) ** (1 / 2)

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return str(self.id)


class Edge:

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.distance = ((start.x - end.x) ** 2 + (start.y - end.y) ** 2) ** (1 / 2)

    def __str__(self):
        return str(self.start.id) + " - " + str(self.end.id)

    def __repr__(self):
        return str(self.start.id) + " - " + str(self.end.id)


class Saving:

    def __init__(self, start, end, saving, a_to_b):
        self.start = start
        self.end = end
        self.saving = saving
        self.a_to_b = a_to_b

    def __str__(self):
        return str(self.start.id) + " - " + str(self.end.id)

    def __repr__(self):
        return str(self.start.id) + " - " + str(self.end.id)


class Route:

    def __init__(self, id_, edges, distancia):
        self.id = id_
        self.reward = 0
        self.edges = edges
        self.distance = distancia
        self.stochastic_of = []
        self.reliability = 0

    def reverse_edges(self):
        edges = []
        for i in range(len(self.edges)):
            edges.append(Edge(self.edges[-(i + 1)].end, self.edges[-(i + 1)].start))

        self.edges = edges

    def copy_edges(self):
        edges = []
        for i in self.edges:
            edges.append(i)

        return edges

    def __str__(self):
        text = str(self.edges[0].start) + "-"
        for i in self.edges[:-1]:
            text += str(i.end) + "-"
        text += str(self.edges[-1].end)
        return text

    def __repr__(self):
        text = str(self.edges[0].start) + "-"
        for i in self.edges[:-1]:
            text += str(i.end) + "-"
        text += str(self.edges[-1].end)
        return text


class BlackBox:
    def __init__(self, beta_0, beta_1, beta_2, beta_3):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.n = len(beta_0)

    def __init__(self, n_type_of_nodes=0):
        self.n = n_type_of_nodes
        self.beta_0 = {i: (random.random() * 2 - 1) / 2 for i in range(n_type_of_nodes)}
        self.beta_1 = [(random.random() * 2 - 1) / 10 for _ in range(n_type_of_nodes)]
        self.beta_2 = {i: (random.random() * 2 - 1) / 2 for i in range(n_type_of_nodes)}
        self.beta_3 = {i: (random.random() * 2 - 1) / 2 for i in range(n_type_of_nodes)}

    def setter_betas(self, beta_0, beta_1, beta_2, beta_3):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.n = len(beta_0)

    def get_value(self, node_type,  weather=0, congestion=0, battery=1):
        weather_value = self.beta_1[node_type] * weather
        congestion_value = self.beta_2[node_type] * congestion
        battery_value = self.beta_3[node_type] * battery
        exponent = self.beta_0[node_type] + weather_value + battery_value + congestion_value
        if exponent < -100:
            exponent = -100

        return 1 / (1 + math.exp(-exponent))


    def simulate(self, node_type, weather=0, congestion=0, battery=0, verbose=False):
        rand = random.random()
        if verbose:
            print("La probabilidad de la black box ha sido: " + str(self.get_value(node_type, weather, congestion, battery)))

        if rand > self.get_value(node_type,  weather, congestion, battery):
            if verbose:
                print("Se pierde el reward del nodo")
            return 0
        else:
            if verbose:
                print("No se pierde el reward del nodo")
            return 1

    def simulate_list(self, list_of_data):
        output = []
        for lista in list_of_data:
            rand = random.random()
            node_type, data = lista[0], lista[1:]
            if rand > self.get_value_with_list(node_type, data):
                output.append(0)
            else:
                output.append(1)

        return output

    def get_value_with_dict(self, dict_of_data):
        node_type = dict_of_data["node_type"]
        battery_value = self.beta_1[node_type] * dict_of_data["battery"]
        weather_value = self.beta_2[node_type] * dict_of_data["weather"]
        congestion_value = self.beta_3[node_type] * dict_of_data["congestion"]
        exponent = self.beta_0[node_type] + weather_value + battery_value + congestion_value
        # Negativo = bueno
        if -exponent > 100:
            exponent = -100
        return 1 / (1 + math.exp(-exponent))

    def simulate_dict(self, dict_of_data):
        output = []
        for list in dict_of_data:
            rand = random.random()
            if rand > self.get_value_with_dict(list):
                output.append(0)
            else:
                output.append(1)

        return output

    def print_beta(self):
        for i in range(self.n):
            print("\n")
            print("Para el tipo de nodo " + str(i) + ":")
            print("Beta_0: " + str(self.beta_0[i]) + " correspondiente al término independiente")
            print("Beta_1: " + str(self.beta_1) + " correspondiente al término que acompaña al battery")
            print("Beta_2: " + str(self.beta_2[i]) + " correspondiente al término que acompaña al weather")
            print("Beta_3: " + str(self.beta_3[i]) + " correspondiente al término que acompaña al congestion")
            print("\n")


class ThompsomSamplingEnvironment:
    def __init__(self):
        self.dict = {}

    def initialize_sampling_dict(self, dict_parameters):
        values = list(dict_parameters.values())
        combinations = list(product(*values))
        self.dict = {combo: [0, 0] for combo in combinations}

    def update_dict(self, key, value):
        if value == 0:
            self.dict[key][1] += 1
        else:
            self.dict[key][0] += 1

    def simulate(self, key):
        s, f = self.dict[key]
        sim = np.random.beta(s + 1, f + 1)
        return sim


if __name__ == "__main__":
    ts = ThompsomSamplingEnvironment()
    ts.initialize_sampling_dict({"whether": [0, 1], "congestion": [0, 1, 3]})
    ts.simulate((0, 1))
    for _ in range(0, 100):
        if np.random.uniform() < 0.7:
            ts.update_dict((0, 1), 1)
        else:
            ts.update_dict((0, 1), 0)
    print((ts.dict))
