from src.utils.read_inputs import read
from src.utils.Constants import *
from src.algorithm.Heuristics import *
from src.algorithm.solution import *
from src.algorithm.static_algorithm import *


def deterministic_approach(max_time):
    nodes = []

    nodes.append(Node(0, 0, 1, 1))
    nodes.append(Node(1, 1, 2, 2))
    nodes.append(Node(2, 2, 1, -1))
    nodes.append(Node(3, 3, 0, 0))
    nodes.append(Node(4, 4, 3, 3))
    nodes.append(Node(5, 0, 4, 4))

    solution2 = Solution(nodes, 10)
    solution2.deterministic_multi_start(max_time)
    solution2.of = np.mean(solution2.routes[0].stochastic_of)
    print("Deterministic in stochastic enviroment: "+ str(solution2.of))


def deterministic_approach_real(max_time):
    nodes, capacity, vehicles = read()
    solution3 = Solution(nodes, capacity, max_vehicles=vehicles)
    solution3.determinstic_algorithm_test()
    print(solution3.routes[0], solution3.of)


def test_dynamic():
    nodes, capacity, vehicles = read()
    solution3 = Solution(nodes, capacity, max_vehicles=vehicles)
    solution3.test_dynamic_algo()


def run_static(nodes, capacity, vehicles, blackbox, iterations):
    s = Static(nodes, capacity, vehicles, bb=blackbox)
    route, static_of, dynamic_of = s.run_multi_start_static(iterations)
    print(route, static_of, dynamic_of)


if __name__ == '__main__':
    nodes, capacity, vehicles = read()
    beta0, beta1, beta2, beta3 = Betas().MEDIUM

    blackbox = BlackBox()
    blackbox.setter_betas(beta0, beta1, beta2, beta3)
    run_static(nodes, capacity, vehicles, blackbox, 100)
    # nodes, max_dist, max_vehicles=1, alpha=0.7, neighbour_limit=-1, dict_of_types=None

    # ts = ThompsomSamplingEnvironment()
    # ts.initialize_sampling_dict({"node_type": [0, 1, 2, 3, 4], "weather": [0, 1], "congestion": [0, 1], "battery": [1]})

    # heur = ConstructiveHeuristic(nodes, capacity, blackbox, ts, weight=0, real_alpha=1, count=0, dict_of_types=None)
    # of_det = heur.test_build_constructive_heuristic_deterministic(10000)
    # of_ts = heur.test_build_constructive_heuristic_ts(10000)

    # print(of_det, of_ts)
