from classes import *
from solution import *
from sim import *
import numpy as np
from read_inputs import read
from Constants import *
from Heuristics import *

# Press the green button in the gutter to run the script.


def deterministic_approach(max_time):
    nodes = []

    nodes.append(node(0, 0, 1, 1))
    nodes.append(node(1, 1, 2, 2))
    nodes.append(node(2, 2, 1, -1))
    nodes.append(node(3, 3, 0, 0))
    nodes.append(node(4, 4, 3, 3))
    nodes.append(node(5, 0, 4, 4))

    solution2 = solution(nodes, 10)
    solution2.deterministic_multi_start(max_time)
    short_simulations = simheuristic(1000, 0.5)
    solution2.routes = [solution2.routes[0]]
    short_simulations.simulation(solution2)
    solution2.of = np.mean(solution2.routes[0].stochastic_of)
    print("Deterministic in stochastic enviroment: "+ str(solution2.of))


def deterministic_approach_real(max_time):
    nodes, capacity, vehicles = read()
    solution3 = solution(nodes, capacity, max_vehicles=vehicles)
    solution3.determinstic_algorithm_test()
    print(solution3.routes[0], solution3.of)
    """

    solution2.deterministic_multi_start(max_time)
    short_simulations = simheuristic(1000, 0.5)
    solution2.routes = [solution2.routes[0]]
    short_simulations.simulation(solution2)
    solution2.of = np.mean(solution2.routes[0].stochastic_of)
    print("Deterministic in stochastic enviroment: "+ str(solution2.of))
    """

def test_dynamic():
    nodes, capacity, vehicles = read()
    solution3 = solution(nodes, capacity, max_vehicles=vehicles)
    solution3.test_dynamic_algo()



if __name__ == '__main__':
    nodes, capacity, vehicles = read()
    beta0, beta1, beta2, beta3 = Betas().MEDIUM

    blackbox = BlackBox()
    blackbox.setter_betas(beta0, beta1, beta2, beta3)

    ts = ThompsomSamplingEnvironment()
    ts.initialize_sampling_dict({"node_type": [0, 1, 2, 3, 4], "weather": [0, 1], "congestion": [0, 1], "battery": [1]})

    heur = ConstructiveHeuristic(nodes,capacity, blackbox, ts, weight=0, real_alpha=1, count=0, dict_of_types=None)
    of_det = heur.test_build_constructive_heuristic_deterministic(10000)
    of_ts = heur.test_build_constructive_heuristic_ts(10000)

    print(of_det, of_ts)

    #deterministic_approach_real(10)
    #Qsimheuristic_approach(10)

    #simheuristic_approach(10)

    #deterministic_approach(10)