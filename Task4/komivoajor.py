'''Task 4_2'''
import random
import math
import matplotlib.pyplot as plt
import numpy as np

def annealing_method_kom():
    # coordinates of cities from the distance matrix
    x = [0.549963E-07, -28.8733, -79.2916, -14.6577, -64.7473, -29.0585, -72.0785, -36.0366,
         -50.4808, -50.5859, -0.135819, -65.0866, -21.4983, -57.5687, -43.0700]

    y = [0.985808E-08, -0.0000000797739, -21.4033, -43.3896, 21.8982, -43.2167, 0.181581,
    -21.6135, 7.37447, -21.5882, -28.7293, -36.0625, 7.31942, -43.2506, 14.5548]


    # distance matrix
    l = np.zeros([15, 15])  # Relative Distance Matrix Template
    for i in np.arange(0, 15, 1):
        for j in np.arange(0, 15, 1):
            if i != j:
                l[i, j] = math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)  # Filling the matrix
            else:
                l[i, j] = float('inf')  # Filling the main diagonal of the matrix


    T = 100 # initial temperature
    alpha = 0.99 # cooling factor
    S = [] # initial random path
    while len(S) != 15:
        num_c = random.randrange(15)
        if num_c not in S:
            S.append(num_c)
    x1 = []
    y1 = []
    for i in range(15):
        x1.append(x[S[i]])
        y1.append(y[S[i]])


    while T > 0.000000001:
        # calculate the length of the path S
        L = 0
        for i in range(15):
            c_2 = i+1
            if c_2 == 15:
                # for closed loop
                c_2 = 0
                # for open loop
                # break

            L += l[S[i]][S[c_2]]


        if T == 100:
            fig, ax = plt.subplots()
            plt.title(f"total path length at the first iteration  {L}")
            ax.plot(x1, y1, 'ro', label='cities')
            ax.plot(x1, y1, label='way')

            plt.grid()
            ax.legend()
            plt.show()

        # swap 2 random cities in vector
        x1 = random.randrange(15)
        x2 = random.randrange(15)
        while x1 == x2:
            x2 = random.randrange(15)

        S1 = S[:]
        S1[x1], S1[x2] = S1[x2], S1[x1]

        # calculate the new path length
        L1 = 0
        for i in range(15):
            c_2 = i+1
            if c_2 == 15:
                # for closed loop
                c_2 = 0
                # for open loop
                # break

            L1 += l[S1[i]][S1[c_2]]

        # transition probability
        if L1 - L < 0:
            S = S1[:]
        else:
            P = 100 * (math.e ** (-((L1 - L) / T)))
            if P > random.randrange(101):
                S = S1[:]

        # decrease in temperature
        T *= alpha

    x1 = []
    y1 = []
    for i in range(15):
        x1.append(x[S[i]])
        y1.append(y[S[i]])

    fig, ax = plt.subplots()
    plt.title(f"total path length at the last iteration  {L}")
    ax.plot(x1, y1, 'ro', label='cities')
    ax.plot(x1, y1, label='way')

    plt.grid()
    ax.legend()
    plt.show()


# if needed, theoretical path is recalculated and drawn
def theory_path():
    x = np.array([0.549963E-07, -28.8733, -79.2916, -14.6577, -64.7473, -29.0585, -72.0785, -36.0366,
             -50.4808, -50.5859, -0.135819, -65.0866, -21.4983, -57.5687, -43.0700])

    y = np.array([0.985808E-08, -0.0000000797739, -21.4033, -43.3896, 21.8982, -43.2167, 0.181581,
    -21.6135, 7.37447, -21.5882, -28.7293, -36.0625, 7.31942, -43.2506, 14.5548])

    indexis_test = np.array([1, 13, 2, 15, 9, 5, 7, 3, 12, 14, 10, 8, 6, 4, 11])
    indexis_test -= 1

    l = np.zeros([15, 15])  # Relative Distance Matrix Template
    for i in np.arange(0, 15, 1):
        for j in np.arange(0, 15, 1):
            if i != j:
                l[i, j] = math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)  # Filling the matrix
            else:
                l[i, j] = float('inf')  # Filling the main diagonal of the matrix


    res_path = 0
    for i in range(14):
        res_path += l[indexis_test[i], indexis_test[i+1]]

    x1, y1 = [], []
    for i in range(15):
        x1.append(x[indexis_test[i]])
        y1.append(y[indexis_test[i]])

    fig, ax = plt.subplots()
    plt.title(f"total path length is {res_path}")
    ax.plot(x, y, 'ro', markersize=6, label='cities')
    ax.plot(x1, y1, 'b')
    # for i in range(1, 16):
    #     ax.annotate(f'{i}', xy=(x[i-1], y[i-1]))
    plt.grid()
    ax.legend()
    plt.show()


annealing_method_kom()
