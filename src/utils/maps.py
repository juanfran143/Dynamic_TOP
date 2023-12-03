import matplotlib.pyplot as plt


class Map:
    def __init__(self, nodes):
        self.nodes = nodes

    def parse_connections(self, connections_str):
        parsed_connections = []

        for connection_str in connections_str:
            # Split the connection string into two nodes
            nodes = connection_str.split('-')

            # Convert nodes to integers and create a tuple
            connection = (int(nodes[0]), int(nodes[1]))

            # Append the tuple to the list
            parsed_connections.append(connection)

        return parsed_connections

    def instance_map(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        for node in self.nodes:
            # Plotting a circle at each coordinate with the reward inside
            circle = plt.Circle((node.x, node.y), 0.5, color='blue', fill=False)
            ax.add_patch(circle)
            ax.annotate(int(node.reward), (node.x, node.y), color='red', ha='center', va='center')

        # Set axis labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Reward Map')

        # Set the limits of the plot based on the nodes
        ax.set_xlim(min(node.x for node in self.nodes) - 1, max(node.x for node in self.nodes) + 1)
        ax.set_ylim(min(node.y for node in self.nodes) - 1, max(node.y for node in self.nodes) + 1)

        ax.set_aspect('equal', adjustable='box')

        # Display the plot
        plt.show()

    def print_route(self, route):
        print(route)
        fig, ax = plt.subplots(figsize=(8, 6))

        for node in self.nodes:
            # Plotting a circle at each coordinate with the reward inside
            if node.reward == 0:
                circle = plt.Circle((node.x, node.y), 0.5, color='red', fill=True)
            else:
                circle = plt.Circle((node.x, node.y), 0.5, color='blue', fill=False)
            ax.add_patch(circle)
            ax.annotate(int(node.id), (node.x, node.y), color='red', ha='center', va='center')

        for team in route:
            for edge in team.edges:
                start_node = edge.start
                end_node = edge.end
                plt.plot([start_node.x, end_node.x], [start_node.y, end_node.y], color='green')


        # Set axis labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Reward Map')

        # Set the limits of the plot based on the nodes
        ax.set_xlim(min(node.x for node in self.nodes) - 1, max(node.x for node in self.nodes) + 1)
        ax.set_ylim(min(node.y for node in self.nodes) - 1, max(node.y for node in self.nodes) + 1)

        ax.set_aspect('equal', adjustable='box')

        # Display the plot
        plt.show()
