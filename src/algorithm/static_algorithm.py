import copy
from src.utils.classes import *
import random as rnd
import numpy as np
import time


class Static:
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
                 dict_of_types=None,
                 max_iter_dynamic=100, select_saving_function=None, random_selection=2):
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

        random.seed = self.seed
        self.weather = random.randint(0, 1)
        self.congestion = {i: random.randint(0, 1) for i in range(len(nodes))}
        self.bb = bb
        self.max_iter_dynamic = max_iter_dynamic

        if dict_of_types:
            self.dict_of_types = dict_of_types
        else:
            self.dict_of_types = {i: 1 for i in range(len(nodes))}

        self.random_selection = random_selection
        self.select_saving_function = select_saving_function

    def reset(self):
        self.routes = []
        self.savings = []

    def dummy_solution(self):
        for i in range(len(self.nodes) - 2):
            edges = [Edge(self.nodes[0], self.nodes[i + 1]), Edge(self.nodes[i + 1], self.nodes[-1])]
            self.routes.append(Route(i, edges, sum([i.distance for i in edges])))
            self.routes[i].reward = self.nodes[i + 1].reward
            self.nodes[i + 1].route = self.routes[i]

    def is_neighbour(self, edge: Edge):
        if self.neighbour_limit == -1:
            return True
        if edge.distance > self.neighbour_limit:
            return False
        return True

    def create_saving_list(self):
        for i in range(len(self.nodes) - 2):
            for j in range(len(self.nodes) - 2):
                if i == j:
                    continue
                edge_a_b = Edge(self.nodes[i + 1], self.nodes[j + 1])
                if not self.is_neighbour(edge_a_b):
                    continue

                edge_a_end = Edge(self.nodes[i + 1], self.nodes[-1])
                edge_depot_b = Edge(self.nodes[0], self.nodes[j + 1])

                saving_distance = self.alpha * (edge_a_end.distance + edge_depot_b.distance - edge_a_b.distance)
                saving_reward = (1 - self.alpha) * (self.nodes[i + 1].reward + self.nodes[j + 1].reward)

                self.savings.append(Saving(self.nodes[j + 1], self.nodes[i + 1], saving_distance + saving_reward,
                                           edge_a_b.distance))

        self.savings.sort(key=lambda x: x.saving)

    def select_saving_grasp(self):
        return self.savings.pop(rnd.randint(0, min([self.random_selection, len(self.savings) - 1])))

    def select_saving_greedy(self):
        return self.savings.pop(0)

    def merge_routes(self, saving):
        route_a = saving.start.route
        route_b = saving.end.route
        distance = route_a.distance + route_b.distance + saving.a_to_b - route_a.edges[-1].distance - route_b.edges[
            0].distance
        if route_a.id != route_b.id and route_a.edges[-1].start.id == saving.start.id and route_b.edges[
            0].end.id == saving.end.id and distance <= self.max_dist:

            route_a.edges.pop()
            route_a.edges = route_a.edges + [Edge(saving.start, saving.end)] + route_b.edges[1:]
            route_a.distance = distance
            route_a.reward += route_b.reward
            for i in route_b.edges[:-1]:
                i.end.route = route_a

            self.routes.pop(self.routes.index(route_b))

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
                        else:
                            original_edge = edges[i].distance + edges[i + 1].distance + edges[j].distance + \
                                            edges[j + 1].distance
                            proposal_edge = y_i.distance(x_i) + y_i.distance(z_j) + y_j.distance(x_i) + \
                                            y_j.distance(z_i)

                        if proposal_edge < original_edge:
                            improve = True
                            edges[i] = Edge(x_i, y_j)
                            edges[i + 1] = Edge(y_j, z_i)
                            edges[j] = Edge(x_j, y_i)
                            edges[j + 1] = Edge(y_i, z_j)
                            route.distance = route.distance + proposal_edge - original_edge

    def calculate_saving_from_last_node(self, last_node, nodes):
        """
        Calculate if possible and return the one with the best "saving"
        :param last_node: Last node visited
        :param nodes: Nodes not used in the solution
        :return:
        """
        end_point = self.nodes[-1]
        saving = {}
        for node in nodes:
            distance = last_node.distance(node)
            if distance + node.distance(end_point) > self.max_dist:
                continue
            saving[node] = distance + node.reward

        if len(saving) == 0:
            return None

        b = list(saving.items())
        b = b.sort(key=lambda x: x[1])
        return b[0][0]

    def local_search_add_nodes(self):
        routes = self.routes[:self.max_vehicles]

        aux = [[i.end.id_ for i in route.edges] for route in routes]
        used_nodes = []
        for i in aux:
            used_nodes += i

        nodes = [i for i in self.nodes if i.id_ not in used_nodes]
        for route in routes:
            improve = True
            while improve:
                improve = False
                last_node = route.edges[-1].start
                selected = self.calculate_saving_from_last_node(last_node, nodes)
                if selected is not None:
                    improve = True
                    nodes.append(selected)
                    route.edges[-1] = Edge(last_node, selected)
                    route.edges.append(Edge(selected, self.nodes[-1]))

    def static_algorithm(self):
        self.dummy_solution()
        self.create_saving_list()
        while len(self.savings) != 0:
            self.merge_routes(self.select_saving_function(self))

        self.routes.sort(key=lambda x: x.reward, reverse=True)
        self.local_search_same_route()
        self.local_search_add_nodes()
        self.of = sum([self.routes[i].reward for i in range(self.max_vehicles)])

        return self.routes, self.of

    def change_environment(self):
        random.seed = self.seed
        self.seed += 1
        self.weather = random.randint(0, 1)
        self.congestion = {i: random.randint(0, 1) for i in range(len(self.nodes))}

    def dynamic_of(self):
        of_list = []
        for _ in range(self.max_iter_dynamic):
            of = 0
            for r in self.routes[:self.max_vehicles]:
                distance = 0
                for e in r.edges[:-1]:
                    self.change_environment()
                    distance += e.distance
                    has_reward = self.bb.simulate(self.dict_of_types[e.end.id], distance / self.max_dist, self.weather,
                                                  self.congestion[e.end.id])
                    of += e.end.reward * has_reward

            of_list.append(of)

        return np.mean(of_list)

    def run_static(self):
        self.static_algorithm()
        dynamic_of = self.dynamic_of()
        return self.routes[:self.max_vehicles], self.of, dynamic_of

    def static_multi_start_iter(self, max_iter: int):
        best_route, best_of = self.static_algorithm()
        for _ in range(max_iter):
            self.reset()
            new_route, new_sol = self.static_algorithm()
            if best_of < new_sol:
                best_of = new_sol
                best_route = copy.deepcopy(new_route)

        self.routes = best_route

    def run_multi_start_static(self, max_iter):
        self.static_multi_start_iter(max_iter)
        dynamic_of = self.dynamic_of()
        return self.routes[:self.max_vehicles], self.of, dynamic_of
