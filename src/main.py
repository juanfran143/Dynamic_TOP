from src.utils.read_inputs import read, read_run
from src.utils.Constants import *
from src.utils.Maps import *

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
    beta0, beta1, beta2, beta3 = Betas().MEDIUM
    instance_dict[Key.BLACKBOX] = BlackBox()
    instance_dict[Key.BLACKBOX].setter_betas(beta0, beta1, beta2, beta3)
    instance_dict[Key.NEIGHBOUR_LIMIT] = neighbour(instance_dict[Key.NODES], instance_dict[Key.PERCENTAGE])
    instance_dict[Key.DICT_OF_TYPE] = {k.id: k.id % instance_dict[Key.N_TYPE_NODES] for k in instance_dict[Key.NODES]}
    list_arguments = [Key.NODES, Key.MAX_DIST, Key.SEED, Key.MAX_VEHICLES, Key.ALPHA, Key.NEIGHBOUR_LIMIT, Key.BLACKBOX,
                      Key.DICT_OF_TYPE, Key.MAX_ITER_DYNAMIC, Key.MAX_ITER_RANDOM]
    filtered_args = {key: instance_dict[key] for key in instance_dict if key in list_arguments}

    return Solution(**filtered_args), instance_dict


if __name__ == '__main__':
    instance = read_run()
    for instance_dict in instance:
        solution, instance_dict = create_instance(instance_dict)

        algo = Algorithm.TYPE_OF_ALGORITH[instance_dict[Key.ALGORITHM]]
        selected_procedure = Algorithm.SELECT_SAVING[instance_dict[Key.SELECTED_NODE_FUNCTION]]
        results = algo(solution, selected_procedure)
        m = Map(instance_dict["nodes"])
        m.print_route(results[0])
