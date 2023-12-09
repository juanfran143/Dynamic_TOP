import copy
from src.utils.classes import *
import numpy as np
from math import log
from src.utils.Constants import Algorithm


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
                 dict_of_types=None, n_types_nodes=2, max_iter_dynamic=100, beta=0.8, select_saving_function=None):
        self.routes = []
        self.of = 0

        self.seed = seed
        random.seed = seed
        np.seed = seed

        self.nodes = nodes
        self.savings = []
        self.alpha = alpha
        self.beta = beta
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

        self.select_saving_function = self.saving_function(select_saving_function)

    def saving_function(self, select_saving_function):
        if select_saving_function == Algorithm.SELECT_SAVING_GREEDY:
            return self.select_saving_greedy
        if select_saving_function == Algorithm.SELECT_SAVING_GRASP:
            return self.get_saving_bias
        if select_saving_function is None:
            return self.get_saving_bias

    def select_saving_greedy(self):
        return self.savings.pop(0)

    def reset(self):
        self.routes = []
        self.savings = []

    def is_neighbour(self, edge: Edge):
        if self.neighbour_limit == -1:
            return True
        if edge.distance > self.neighbour_limit:
            return False
        return True

    def get_saving_greedy(self):
        return self.savings.pop(0)

    def get_saving_bias(self):
        index = int((log(np.random.random()) / log(1 - self.beta))) % len(self.savings)
        return self.savings.pop(index)

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
                    saving = self.select_saving_function()
                    self.routes[v].append(saving.end)
                    nodes_used.append(saving.end.id)

                    reward[v] += saving.end.reward
                    dist[v] += saving.a_to_b
                    node_id = saving.end.id
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
                    dynamic_reward[v] += saving.end.reward * has_reward
                else:
                    last = Edge(self.routes[v][-1], self.nodes[-1])
                    self.routes[v].append(self.nodes[-1])
                    dist[v] += last.distance

        routes = []
        for k, v in self.routes.items():
            edges = []
            for n in range(len(v)-1):
                edges.append(Edge(v[n], v[n+1]))
            routes.append(Route(k, edges, dist[k]))
            routes[k].reward = reward[k]
        self.routes = routes
        return sum(dynamic_reward[v] for v in range(self.max_vehicles))

    def local_search_same_route(self):
        """
        En el determinista no tiene sentido, pero en el simheurístico sí.
        :return:
        """
        # 0-6-30-17-19-32
        for route in self.routes:
            improve = True
            while improve:
                improve = False
                edges = route.edges
                for i in range(len(route.edges) - 2):
                    for j in range(i + 1, len(route.edges) - 1):
                        x_i, y_i, z_i = edges[i].start, edges[i].end, edges[i + 1].end
                        x_j, y_j, z_j = edges[j].start, edges[j].end, edges[j + 1].end
                        if j == i + 1:
                            original_edge = edges[i].distance + edges[j + 1].distance
                            proposal_edge = y_i.distance(z_j) + y_j.distance(x_i)

                            if round(proposal_edge, 2) < round(original_edge, 2):
                                improve = True
                                edges[i] = Edge(x_i, y_j)
                                edges[i + 1] = Edge(y_j, y_i)
                                edges[j + 1] = Edge(y_i, z_j)
                                route.distance = route.distance + proposal_edge - original_edge

                        else:
                            original_edge = edges[i].distance + edges[i + 1].distance + edges[j].distance + \
                                            edges[j + 1].distance
                            proposal_edge = y_i.distance(x_j) + y_i.distance(z_j) + y_j.distance(x_i) + \
                                            y_j.distance(z_i)

                            if round(proposal_edge, 2) < round(original_edge, 2):
                                improve = True
                                edges[i] = Edge(x_i, y_j)
                                edges[i + 1] = Edge(y_j, z_i)
                                edges[j] = Edge(x_j, y_i)
                                edges[j + 1] = Edge(y_i, z_j)
                                route.distance = route.distance + proposal_edge - original_edge

    def calculate_saving_from_last_node(self, last_node, nodes, current_distance):
        """
        Calculate if possible and return the one with the best "saving"
        :param current_distance:
        :param last_node: Last node visited
        :param nodes: Nodes not used in the solution
        :return:
        """
        end_point = self.nodes[-1]
        saving = {}
        for node in nodes:
            distance = last_node.distance(node)
            if distance + node.distance(end_point) + current_distance > self.max_dist:
                continue
            saving[node] = self.alpha * distance + (1-self.alpha) * node.reward

        if len(saving) == 0:
            return None

        sorted_d = dict(sorted(saving.items(), key=lambda item: item[1], reverse=True))
        first_element = next(iter(sorted_d.items()))
        return first_element[0]

    def local_search_add_nodes(self):
        routes = self.routes[:min(self.max_vehicles, len(self.routes))]

        used_nodes = [[i.end.id for i in route.edges] for route in routes][0]
        for route in routes:
            improve = True
            while improve:
                improve = False
                last_node = route.edges[-1].start
                nodes = [i for i in self.nodes[1:-1] if i.id not in used_nodes]
                selected = self.calculate_saving_from_last_node(last_node, nodes, route.distance)
                if selected is not None:
                    improve = True
                    used_nodes.append(selected.id)
                    new_node = Edge(last_node, selected)
                    return_home = Edge(selected, self.nodes[-1])
                    route.distance = (route.distance - route.edges[-1].distance + new_node.distance +
                                      return_home.distance)
                    route.edges[-1] = new_node
                    route.edges.append(return_home)

    def change_environment(self):
        random.seed = self.seed
        self.seed += 1
        self.weather = random.choice([-1, 1])
        self.congestion = {i: random.choice([-1, 1])for i in range(len(self.nodes))}

    def change_seed(self):
        self.seed += 10000
        random.seed = self.seed
        np.seed = self.seed

    def dynamic_of(self):
        dynamic_of = 0
        for r in self.routes[:min(self.max_vehicles, len(self.routes))]:
            distance = 0
            for e in r.edges[:-1]:
                distance += e.distance
                has_reward = self.bb.simulate(self.dict_of_types[e.end.id], distance / self.max_dist, self.weather,
                                              self.congestion[e.end.id])
                dynamic_of += e.end.reward * has_reward
                self.change_environment()

        return dynamic_of

    def static_solution(self, max_iter):
        best_of = -1
        best_route = []
        for _ in range(max_iter):
            self.constructive_static_solution()
            self.local_search_same_route()
            self.local_search_add_nodes()
            of_list = sum([self.routes[i].reward for i in range(self.max_vehicles)])
            if best_of < of_list:
                best_of = of_list
                best_route = copy.deepcopy(self.routes)

        self.routes = best_route

    def run_static_constructive(self, max_iter):
        of_dynamic_list, of_list, route_list = [], [], []
        for _ in range(self.max_iter_dynamic):
            self.static_solution(max_iter)
            of_dynamic_list.append(self.dynamic_of())
            of_list.append(sum([self.routes[i].reward for i in range(self.max_vehicles)]))
            route_list.append(copy.deepcopy(self.routes))
            self.change_seed()

        return route_list, of_list, of_dynamic_list
