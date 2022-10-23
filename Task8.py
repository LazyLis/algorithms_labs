import numpy as np
import networkx as nx
import time
import random as rd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


# check whether there is a requirement for graph type in edmonds_karp method
def check_graph_types(draw_graph=True):
    G = nx.DiGraph()
    G.add_edge(1, 2, capacity=3.0, weight=1.0)
    G.add_edge(1, 3, capacity=2.0, weight=2.0)
    G.add_edge(3, 4, capacity=1.0, weight=2.5)
    G.add_edge(2, 4, capacity=5.0, weight=3.0)
    G.add_edge(2, 5, capacity=4.0, weight=5.0)
    G.add_edge(5, 6, capacity=6.0, weight=6.0)
    G.add_edge(4, 2, capacity=2.0, weight=2.0)
    G.add_edge(4, 4, capacity=3.0, weight=1.0)
    G.add_node(7)
    G.add_edge(8, 9, capacity=3.0, weight=1.0)

    flow_value, flow_dict = nx.maximum_flow(G, 7, 4, flow_func=nx.algorithms.flow.edmonds_karp)
    print(flow_value)
    if draw_graph:
        pos = {1: [1, 5], 2: [3, 9], 3: [3, 1], 4: [6, 0], 5: [6, 9], 6: [10, 5], 7: [7, 4], 8: [11, 6], 9: [11, 9]}
        nx.draw_networkx(G, node_size=500, arrows=True, arrowsize=15, pos=pos)
        plt.show()


# generate graph of random structure with both capacity and weight features of edges
def generate_graph(n_vert, n_edges):
    G = nx.gnm_random_graph(n_vert, n_edges)

    for (u, v, w) in G.edges(data=True):
        w['weight'] = rd.randint(1, 50)
        w['capacity'] = u + v

    return G


# implement and count time needed to run edmonds_karp method
def edmonds_karp(G):
    s, t = 0, 0
    # nodes_ls = G.nodes
    while s == t:
        s = rd.randint(0, len(G.nodes)-1)
        t = rd.randint(0, len(G.nodes)-1)
    start = time.time()
    nx.maximum_flow(G, s, t, flow_func=nx.algorithms.flow.edmonds_karp)
    end = time.time()
    return end - start


# implement and count time needed to run floyd_warshall method
def warshall(G):
    start = time.time()
    nx.floyd_warshall(G, weight='weight')
    end = time.time()
    return end - start


# calculate complexities for all graphs
def estimate_complexity(graphs_ls, nodes_n, edges_n):
    edmonds_karp_time = []
    warshall_time = []

    for i in range(len(graphs_ls)):
        edmonds_karp_time.append(edmonds_karp(graphs_ls[i]))
        warshall_time.append(warshall(graphs_ls[i]))

    theor_edmonds = theoretical_complexity(nodes_n, edges_n, edmonds_karp_time)
    theor_warshall = theoretical_complexity(nodes_n, edges_n, warshall_time, type='warshall')
    return (theor_edmonds, edmonds_karp_time), (theor_warshall, warshall_time)


# objective functions for least_squares
def func(c, g, f):
    return c[0] * f + c[1] - g
def func1(c, g, f):
    return c * f - g


# calculate theorerical complexities for all graphs
def theoretical_complexity(nodes_n, edges_n, emp_time, type='edmond'):
    def warshall_theory(node, edge):
        return node * node * node

    def edmond_theory(node, edge):
        return node * edge * edge

    theory = edmond_theory if type == 'edmond' else warshall_theory

    theor_ls = []
    for i in range(len(nodes_n)):
        theor_ls.append(theory(nodes_n[i], edges_n[i]))
    emp = np.array(emp_time)
    teor = np.array(theor_ls)

    for i in range(len(emp)):
        if emp[i] > 0.025:
            if i >= len(emp)-1:
                emp[i] = emp[i - 1] / 2.
            else: emp[i] = (emp[i-1] + emp[i+1]) / 2.

    # res = least_squares(func, [0.1, 0.1], method='lm', xtol=1e-3, args=(emp, teor), verbose=1)
    # teor_new = [res.x[0] * time_ + res.x[1] for time_ in teor] #res.x

    res = least_squares(func1, 0.1, method='lm', xtol=1e-3, args=(emp, teor), verbose=1)
    teor_new = [res.x * time_ for time_ in teor]
    return teor_new


# visualize obtained complexities
def visualize(teor, emp, name, number):
    fig, ax = plt.subplots()
    ax.plot(number, emp, label='Experimental results')
    ax.plot(number, teor, label='Approximation based on theoretical estimates')
    # ax.set_title(f'{name}', fontsize=24)
    ax.set_xlabel(f'Number of {name}', fontsize=14)
    ax.set_ylabel('Time (seconds)', fontsize=14)
    plt.grid()
    ax.legend()
    plt.show()


# initialize nodes and edges list for floyd_warshall method
def make_node_edges_ls():
    nodes_n = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    edges_n = [50, 60, 70, 80, 90, 100, 110, 130, 150, 180, 200]

    ls_node1 = [x for x in range(21, 41)]
    ls_edges1 = [200 + 15 * i for i in range(len(ls_node1))]

    ls_node2 = [x for x in range(41, 51)]
    ls_edges2 = [500 + 20 * i for i in range(len(ls_node2))]

    ls_node3 = [x for x in range(51, 71)]
    ls_edges3 = [700 + 10 * i for i in range(len(ls_node3))]

    ls_node4 = [x for x in range(71, 91)]
    ls_edges4 = [900 + 5 * i for i in range(len(ls_node4))]

    return nodes_n + ls_node1 + ls_node2 + ls_node3 + ls_node4, \
           edges_n + ls_edges1 + ls_edges2 + ls_edges3 + ls_edges4


# check_graph_types(draw_graph=False)

'''edmonds init '''
'''check time complexity by vertices'''
# nodes_n = [i for i in range(60, 100)]
# edges_n = [3500] * len(nodes_n)

'''check time complexity by edges'''
# edges_n = [i for i in range(100, 4001, 100)]
# nodes_n = [100] * len(edges_n)

''' warshall floyd init'''
nodes_n, edges_n = make_node_edges_ls()
graph_ls = []

for i in range(len(nodes_n)):
    graph_ls.append(generate_graph(nodes_n[i], edges_n[i]))

(theor_edmonds, edmonds_emp), (theor_warshall, warshall_emp) = estimate_complexity(graph_ls, nodes_n, edges_n)

visualize(theor_warshall, warshall_emp, "vertices", nodes_n)

# visualize(theor_edmonds, edmonds_emp, "vertices", nodes_n)
# visualize(theor_edmonds, edmonds_emp, "edges", edges_n)
