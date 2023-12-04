import copy
from src.utils.classes import *
import numpy as np


class StaticConstructive:
    """
    Clase que ejecutará el algoritmo statico, es decir, el que no tiene en cuenta
    el dinamismo del problema. Habrá una función statica que es la que se tendrá
    que ejecutar: "static_algorithm(nodes, max_dist, max_vehicles, alpha, bb, dict_of_types, neighbour_limit=inf)".

    Esta función tiene como input los conceptos básicos:
        - nodes: Nodos de la instancia
        - max_distance: Máxima distancia que puede hacer los drones
        - max_vehicles: Número de drones
        - alpha: Parámetro que regula los savings
        - bb: Black-box de nuestro problema
        - dict_of_types: Los diferetes tipos de nodos que tenemos

    Opcional:
        - neighbour_limit: Marcará el límite en distancia que puede haber entre 2 nodos para marcarlos como vecinos.
                           Por defecto será infinito.

    Devuelve:
        - Función objetivo media de los escenarios estáticos
        - Función objetivo media de los escenarios dinámica
        - Ruta para cada instancia

    """

    def __init__(self, nodes, max_dist, seed=0, max_vehicles=1, alpha=0.7, neighbour_limit=-1, bb=None,
                 dict_of_types=None, n_types_nodes=2, max_iter_dynamic=100):
        self.routes = []
        self.of = 0

        self.seed = seed
        random.seed = seed
        np.seed = seed

        self.nodes = nodes
        self.savings = []
        self.alpha = alpha
        self.max_dist = max_dist
        self.max_vehicles = max_vehicles
        self.neighbour_limit = neighbour_limit

        self.n_types_nodes = n_types_nodes

        random.seed = self.seed
        self.weather = random.choice([-1, 1])
        self.congestion = {i: random.choice([-1, 1]) for i in range(len(nodes))}
        self.bb = bb
        self.new_data = {i: [] for i in range(n_types_nodes)}
        self.max_iter_dynamic = max_iter_dynamic

        if dict_of_types:
            self.dict_of_types = dict_of_types
        else:
            self.dict_of_types = {i: 1 for i in range(len(nodes))}

    def reset(self):
        self.routes = []
        self.savings = []

    def is_neighbour(self, edge: Edge):
        if self.neighbour_limit == -1:
            return True
        if edge.distance > self.neighbour_limit:
            return False
        return True

    def constructive_static_solution(self):
        self.routes = {}
        self.new_data = {i: [] for i in range(self.n_types_nodes)}
        nodes = copy.deepcopy(self.nodes)

        self.routes = {k: [nodes[0]] for k in range(self.max_vehicles)}
        end = {k: False for k in range(self.max_vehicles)}
        dist = {k: 0 for k in range(self.max_vehicles)}
        reward = {k: 0 for k in range(self.max_vehicles)}
        dynamic_reward = {k: 0 for k in range(self.max_vehicles)}
        nodes_used = []
        while sum(end.values()) != len(end):
            for v in range(self.max_vehicles):
                if end[v]:
                    continue
                start_node = self.routes[v][-1]
                self.savings = []
                for j in range(len(self.nodes) - 2):
                    edge_a_b = Edge(start_node, self.nodes[j + 1])

                    if not self.is_neighbour(edge_a_b) or self.nodes[j+1].id in nodes_used:
                        continue

                    edge_depot_b = Edge(self.nodes[j + 1], self.nodes[-1])

                    if dist[v] + edge_depot_b.distance + edge_a_b.distance > self.max_dist:
                        continue

                    saving_distance = self.alpha * edge_a_b.distance
                    saving_reward = (1 - self.alpha) * self.nodes[j + 1].reward

                    self.savings.append(Saving(start_node, self.nodes[j + 1], saving_distance + saving_reward,
                                               edge_a_b.distance))

                if len(self.savings) == 0:
                    end[v] = True
                else:
                    self.savings.sort(key=lambda x: x.saving)

                if len(self.savings) != 0:
                    self.routes[v].append(self.savings[0].end)
                    nodes_used.append(self.savings[0].end.id)

                    reward[v] += self.savings[0].end.reward
                    dist[v] += self.savings[0].a_to_b
                    node_id = self.savings[0].end.id
                    node_type = self.dict_of_types[node_id]

                    weather = self.weather
                    congestion = self.congestion[node_id]
                    # transformed into (-1,1)
                    battery = ((1-dist[v]/self.max_dist) - 0.5) * 2
                    # print(weather,congestion,battery)
                    has_reward = self.bb.simulate(node_type=node_type, weather=weather,
                                                  congestion=congestion, battery=battery, verbose=False)
                    new_row = np.array([weather, congestion, battery, has_reward])
                    self.new_data[node_type].append(new_row)
                    dynamic_reward[v] += self.savings[0].end.reward * has_reward
                else:
                    last = Edge(self.routes[v][-1], self.nodes[-1])
                    self.routes[v].append(self.nodes[-1])
                    dist[v] += last.distance
                # TODO: Meter LS
                self.change_environment()

        routes = []
        for k, v in self.routes.items():
            edges = []
            for n in range(len(v)-1):
                edges.append(Edge(v[n], v[n+1]))
            routes.append(Route(k, edges, dist[k]))
            routes[k].reward = reward[k]
        self.routes = routes
        return sum(dynamic_reward[v] for v in range(self.max_vehicles))

    def change_environment(self):
        random.seed = self.seed
        self.seed += 1
        self.weather = random.choice([-1, 1])
        self.congestion = {i: random.choice([-1, 1])for i in range(len(self.nodes))}

    def change_seed(self):
        self.seed += random.randint(1000, 10000)
        random.seed = self.seed
        np.seed = self.seed

    def run_static_constructive(self):
        of_dynamic_list, of_list, route_list = [], [], []
        for _ in range(self.max_iter_dynamic):
            of_dynamic_list.append(self.constructive_static_solution())
            of_list.append(sum([self.routes[i].reward for i in range(self.max_vehicles)]))
            route_list.append(copy.deepcopy(self.routes))
            self.change_seed()

        return route_list, of_list, of_dynamic_list