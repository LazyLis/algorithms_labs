import itertools
# importing a library for working with graphs
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(27) # setting the seed of randomness
# 5.1
# creating a random incidence matrix
def adjacency_matrix(nodes, edges):
    G = np.zeros((nodes, nodes))
    list = []
    while edges:
        i = random.randint(0, 99)
        j = random.randint(0, 99)
        list.append([i, j])
        list.append([j, i])
        if G[i][j] != 1 and G[j][i] != 1 and j != i:
            G[i][j], G[j][i] = 1, 1 # matrix symmetry condition
            edges -= 1
        else:
            continue
    return G

# creating an incident list from a matrix
def adjacency_from_matrix(G_matrix):

    G = nx.Graph()
    G.add_nodes_from([i for i in range(len(G_matrix))])

    def get_indices(nod, el=1):
        list = []
        for i in range(len(nod)):
            if nod[i] == el:
                list.append(i)
        return list


    for i in range(len(G_matrix)):
        edges = get_indices(G_matrix[i])
        for edge in edges:
            G.add_edge(i,edge)

    return G



G_adj_matrix = adjacency_matrix(100, 200)
G_adj_list = adjacency_from_matrix(G_adj_matrix)

# 5.2

def DFS(start_node):
    return list(nx.dfs_edges(G_adj_list, source=start_node))


def BFS(start_node, stop_node):
    # perform a breadth-first crawl using the library function
    pairs = list(nx.bfs_edges(G_adj_list, stop_node))
    # the path list
    path = [start_node]

    for i in range(len(pairs) - 1, -1, -1):
        if pairs[i][1] == start_node:
            start_node = pairs[i][0]
            path.append(start_node)

    return np.array(path)


np.random.seed(26)
node_1, node_2 = np.random.randint(0, 100, 2)

# results for the first part of the task
print(f"first 3 rows of adjacency matrix\n"
      f"{G_adj_matrix[0]}\n"
      f"{G_adj_matrix[1]}\n"
      f"{G_adj_matrix[0]}\n\n"
      f"first 3 lines of the adjacency list\n"
      f"{G_adj_list.edges(0)}\n"
      f"{G_adj_list.edges(1)}\n"
      f"{G_adj_list.edges(2)}\n")

# results for the second part of the task
print(f"Selected nodes: {node_1}, {node_2} \n\n"
      f"the first elements of the list of graph connectivity components found using DFS: \n"
      f"{format(DFS(node_1)[0])} "
      f"{format(DFS(node_1)[1])} "
      f"{format(DFS(node_1)[2])} "
      f"{format(DFS(node_1)[3])}\n "
      f"\nThe path between two randomly selected nodes of the graph \n"
      f"{' - '.join(str(n) for n in BFS(start_node=node_1, stop_node=node_2))}")

# graphically displaying the graph
nx.draw_kamada_kawai(G_adj_list, node_size = 15)
plt.show()


