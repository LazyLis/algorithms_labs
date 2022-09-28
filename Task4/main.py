''' Task4_1 '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import least_squares
from particle_swarm_optimization import PSO
import random
import math


# x and y initialization
def init(plot=False):
    x = np.array([3 * k / 1000 for k in range(1001)])
    y = np.array(len(x) * [0.0])

    np.random.seed(1)
    noise = np.random.normal(0., 1., size=len(x))

    for i in range(len(x)):
        f_x = 1 / (x[i]**2 - 3*x[i] + 2)
        if f_x < -100:
            y[i] = -100 + noise[i]
        elif f_x > 100:
            y[i] = 100 + noise[i]
        else:
            y[i] = f_x + noise[i]

    if plot == True:
        plt.plot(x, y)
        plt.grid()
        plt.show()
    return x, y

def F(x, c):
    return (c[0] * x + c[1]) / (x*x + c[2] * x + c[3])

def D_abcd(c, F, x, y):

    D_ab = (F(x, c) - y)
    return D_ab

def D_abcd2(c, F, x, y):

    D_ab = sum((F(x, c) - y)**2)
    return D_ab

# Nelder-Mead method
def nelder_method(x, y, c0):

    res = minimize(D_abcd2, c0, args=(F, x, y), method='nelder-mead',
                   options={"xatol": 1e-3})

    print('\n')
    print(f"Nelder-Mead\n"
          f"x: {res['x']}\n"
          f"f(x) = {res['fun']}\n"
          f"Number of iterations: {res['nit']}\n"
          f"Function evaluations: {res['nfev']}\n")
    return res['x']


# Levenberg-Marquardt method
def levenberg_marquardt(x, y, c0):

    # method='lm' requires object function that is not a scalar (it has it dot product inside the least_squares method)
    res = least_squares(D_abcd, c0, method='lm', xtol=1e-3, args=(F, x, y), verbose=1)
    print('\n')
    print(f"Levenberg-Marquardt:\n"
          f"x: {res.x}\n"
          f"f(x) = {D_abcd2(res.x, F, x, y)}\n"
          # f"number of iterations: {res['nit']}\n"
          f"Function evaluations: {res.nfev}\n")
    return res.x


# Particle Swarm Optimization
def particle_swarm(x, y):
    pso = PSO(population_size=20, max_steps=1000, args=(x, y))
    # (600, 150) w=0.05, 207801 (49 iter); (600, 150), 0.5 - 199038
    c_PSO, PSO_info = pso.evolve(silent=True)
    print('\n')
    print(f"Particle swarm optimization:\n"
          f"x: {c_PSO}\n"
          f"f(x) = {D_abcd2(c_PSO, F, x, y)}\n"
          f"Number of iterations: {PSO_info['nit']}\n"
          f"Function evaluations: {PSO_info['nfev']}\n")
    return c_PSO


# Simulated annealing
def annealing_method(x_init, y_init):

    def aimFunction(x, y, a, b, c, d):
        f = (a * x + b) / (x*x + c * x + d)
        D_ab = sum((f - y) ** 2)
        return D_ab

    # x_init, y_init = init()
    T = 10  # initiate temperature
    Tmin = 0.001  # minimum value of terperature
    alpha = 0.99
    num_iter = 0
    num_calc = 0
    np.random.seed()
    a = np.random.uniform(low=-3, high=2)  # initiate x
    b = np.random.uniform(low=-3, high=2)
    c = np.random.uniform(low=-3, high=2)
    d = np.random.uniform(low=-3, high=2)
    k = 10  # times of internal circulation
    y = 0  # initiate result
    step_min = -1.1
    step_max = 1.1
    while T >= Tmin:
        num_iter += 1
        # calculate y
        y = aimFunction(x_init, y_init, a, b, c, d)
        num_calc += 1
        # generate a new x in the neighboorhood of x by transform function
        aNew = a + np.random.uniform(low=step_min, high=step_max) * T
        bNew = b + np.random.uniform(low=step_min, high=step_max) * T
        cNew = c + np.random.uniform(low=step_min, high=step_max) * T
        dNew = d + np.random.uniform(low=step_min, high=step_max) * T

        if (-4 <= aNew and aNew <= 3) and (-4 <= bNew and bNew <= 3) and \
                (-4 <= cNew and cNew <= 3) and (-4 <= dNew and dNew <= 3):
            yNew = aimFunction(x_init, y_init, aNew, bNew, cNew, dNew)
            num_calc += 1
            if yNew - y < 0:
                a = aNew
                b = bNew
                c = cNew
                d = dNew
            else:
                # metropolis principle
                p = math.exp(-(yNew - y) / T)
                r = np.random.uniform(low=0, high=1)
                if r < p:
                    a = aNew
                r = np.random.uniform(low=0, high=1)
                if r < p:
                    b = bNew
                r = np.random.uniform(low=0, high=1)
                if r < p:
                    c = cNew
                r = np.random.uniform(low=0, high=1)
                if r < p:
                    d = dNew
        T = T * alpha
    print(f"annealing method,\n"
          f"a, b, c, d: {a, b, c, d}\n"
          f"f(x) = {aimFunction(x_init, y_init, a, b, c, d)}\n"
          f"number of iterations: {num_iter}\n"
          f"number of function calculations: {num_calc}\n")
    return a, b, c, d


x, y = init()
c0 = np.array([1, 1, 1, 1])
c_NM = nelder_method(x, y, c0)
c_LM = levenberg_marquardt(x, y, c0)
c_PSO = particle_swarm(x, y)
a, b, c, d = annealing_method(x, y)

fig, ax = plt.subplots()
plt.title('Unconstrained nonlinear optimization')

ax.plot(x, y, 'ro', markersize=1, label='Generated data')
ax.plot(x, F(x, c_NM), label='Nelder-Mead method', color='magenta')
ax.plot(x, F(x, c_PSO), label='Particle swarm algorithm', color='y')
ax.plot(x, F(x, c_LM), label='Levenberg-Marquardt', color='g')
ax.plot(x, (a * x + b) / (x * x + c * x + d), label='Annealing method', color='b')

plt.grid()
ax.legend()
plt.show()
