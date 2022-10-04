import numpy as np
import networkx as nx
import time
import random as rd
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
from pyvis.network import Network
from decimal import Decimal

''' Functions for Task 6_1 '''

#  Create adjacency matrix
def adjacency_mx(vertices, edges):
    np.random.seed(1)
    ls_pairs = []

    for i in range(vertices-1):
        for j in range(i+1, vertices):
            ls_pairs.append((i, j))

    pairs_idx = [i for i in range(len(ls_pairs))]
    pairs_choice = np.random.choice(pairs_idx, size=edges, replace=False)

    mx = np.zeros(shape=(vertices, vertices))
    factor = np.random.randint(low=1, high=100, size=len(pairs_choice))
    for i in range(len(pairs_choice)):
        mx[ls_pairs[pairs_choice[i]][0], ls_pairs[pairs_choice[i]][1]] = factor[i]
        mx[ls_pairs[pairs_choice[i]][1], ls_pairs[pairs_choice[i]][0]] = factor[i]

    return mx


# Implement Dijkstra's algorithm
def dijkstra(graph, start_node):

    ls_paths = []
    for final_node in graph.nodes:
        if final_node != start_node:
            ls_paths.append(nx.dijkstra_path(graph, start_node, final_node))

    return ls_paths


# Implement Bellman-Ford's algorithm
def bellman_ford(graph, start_node):

    ls_paths = []
    for final_node in graph.nodes:
        if final_node != start_node:
            ls_paths.append(nx.bellman_ford_path(graph, start_node, final_node))

    return ls_paths


# Mainfunction for task 6-1
def mean_time_dijkstra_bellman(draw_graph=False):

    adj_mx = adjacency_mx(vertices=100, edges=500)
    graph = nx.Graph(adj_mx)
    if draw_graph:
        nx.draw(graph, node_size=50)
        plt.show()

    rd.seed(1)
    start_node = rd.randint(0, len(graph) - 1)

    dijkstra_times, bellman_times = [], []
    for i in range(10):
        start_time_dijkstra = time.time()
        dijkstra(graph, start_node)
        # nx.dijkstra_path(graph, start_node, start_node+1 if start_node+1 <= len(graph) - 1 else start_node-1)
        dijkstra_times.append(time.time() - start_time_dijkstra)

        start_time_bellman = time.time()
        bellman_ford(graph, start_node)
        bellman_times.append(time.time() - start_time_bellman)

    print('Mean time of Dijkstra algorithm, seconds: ' + str(np.mean(dijkstra_times)))
    print('Mean time of Bellman-Ford algorithm, seconds: ' + str(np.mean(bellman_times)))


''' Functions for Task 6_2 '''


# Generating cell grid
def generate_cells_mx(rows=10, cols=20, obstacle_num=40, visualize=False):
    np.random.seed(1)
    cell_idx = [i for i in range(rows*cols)]
    arr_cell = np.array([0] * rows*cols)

    obstacle_idx = np.random.choice(cell_idx, size=obstacle_num, replace=False)
    for idx in obstacle_idx:
        arr_cell[idx] = 1
    mx_cell = np.reshape(arr_cell, (rows, cols))

    if visualize:
        fig, ax = plt.subplots()
        ax.matshow(mx_cell, cmap=plt.cm.Blues)
        for j in range(cols):
            for i in range(rows):
                c = i*cols + j
                ax.text(j, i, str(c), va='center', ha='center')

        ax.set_xticks(np.arange(-0.5, cols-0.5, 1.))
        ax.set_yticks(np.arange(-0.5, rows-0.5, 1.))
        ax.grid()
        plt.show()
    return mx_cell, obstacle_idx


# Constructing an adjacency matrix by the given cell grid
def adj_cells_mx(rows=10, cols=20, visualize_cells=False):
    mx_cells, obstacle_idx = generate_cells_mx(visualize=visualize_cells)
    obstacle_idx.sort()

    adj_mx = np.zeros((rows*cols, cols*rows))
    # make edges between all cells
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            v1 = i*cols + j
            # vertices above and below, to the left and right
            for add in [-cols, cols, -1, 1]:
                adj_mx[v1, v1 + add] = 1
                adj_mx[v1 + add, v1] = 1

    for i in (0, rows-1):
        for j in range(1, cols-1):
            v1 = i * cols + j
            # vertices to the left and right
            for add in [-1, 1]:
                adj_mx[v1, v1 + add] = 1
                adj_mx[v1 + add, v1] = 1

    for i in range(1, rows-1):
        for j in (0, cols-1):
            v1 = i*cols + j
            # vertices above and below
            for add in [-cols, cols]:
                adj_mx[v1, v1 + add] = 1
                adj_mx[v1 + add, v1] = 1

    # exclude edges, that lead to an obstacle
    for v1 in obstacle_idx:
        for add in [-cols, cols, -1, 1]:
            if 0 <= v1 + add < rows * cols:
                adj_mx[v1, v1 + add] = 0
                adj_mx[v1 + add, v1] = 0

    return adj_mx, obstacle_idx


# Implementing A* algorithm
def a_asterix(graph, obstacle_idx, custom_obstacle_vertices):
    np.random.seed()
    set_vertices = set(graph.nodes)
    set_obstacle = set(list(obstacle_idx) + custom_obstacle_vertices)
    available_vert = list(set_vertices-set_obstacle)
    nodes = np.random.choice(available_vert, size=2, replace=False)

    path = nx.astar_path(graph, nodes[0], nodes[1])
    return path, nodes


# Paint the path in red color
def set_path_color(path, graph):
    color_map = []
    for _ in graph.nodes:
        color_map.append('tab:blue')

    for edge in graph.edges:
        graph[edge[0]][edge[1]]['color'] = "tab:blue"

    for i in range(len(path)-1):
        graph[path[i]][path[i+1]]['color'] = "tab:red"
        color_map[path[i]] = "tab:red"

    color_map[path[-1]] = "tab:red"
    return color_map


# Draw the path in the graph
def draw_path_in_graph(graph, color_map, cols=20):
    nodes = graph.nodes
    pos = {n: [n // cols, n - (n // cols) * cols] for n in nodes}
    nx.draw_networkx(graph, pos=pos, edge_color=nx.get_edge_attributes(graph, 'color').values(),
                                     node_color=color_map)
    plt.show()


# The main function for the task 6-2
def time_shortest_cells(draw_graph=False, visualize_cells=False, visualize_paths=False):
    adj_mx, obs_idx = adj_cells_mx(visualize_cells=visualize_cells)
    graph = nx.Graph(adj_mx)

    if draw_graph:
        nx.draw(graph)
        nt = Network('890px', '1200px')
        nt.from_nx(graph)
        nt.show('nx.html')

    ls_times_a = []
    for _ in range(5):
        start_time = time.time()
        path, nodes = a_asterix(graph, obs_idx, [19, 39])
        now = time.time()
        ls_times_a.append(now-start_time)

        if visualize_paths:
            color_map = set_path_color(path, graph)
            draw_path_in_graph(graph, color_map)

        print("Start node:", nodes[0], "end node:", nodes[1])
        print(path, '\n')

    print("List of measured time:", ls_times_a)


''' Task 6_1 '''
# mean_time_dijkstra_bellman(draw_graph=False)

''' Task 6_2 '''
time_shortest_cells(draw_graph=False, visualize_cells=False, visualize_paths=False)
