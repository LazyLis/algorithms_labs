import numpy as np
import matplotlib.pyplot as plt


class PSO(object):
    def __init__(self, population_size, max_steps, args=()):
        self.args = args
        self.c1 = self.c2 = 2
        self.population_size = population_size  # number of particles
        self.dim = 4
        self.w = np.array([0.6] * self.dim)  # weights
        self.max_steps = max_steps  # number of iterations
        self.c_bound = [-5, 5]  # interval for search of a solution

        np.random.seed(1)
        self.c = np.random.uniform(self.c_bound[0], self.c_bound[1],
                                   (self.population_size, self.dim))  # initial position of particles
        self.v = np.random.rand(self.population_size, self.dim)  # initial velocity of particles


        fitness = self.calculate_fitness(self.c)
        self.p = self.c  # the best position of a particle
        self.pg = self.c[np.argmin(fitness)]  # the best position among the particles
        self.individual_best_fitness = fitness  # the best objective function for a particle
        self.global_best_fitness = np.max(fitness)  # the best objective function among the particles

    # calculating the objective function
    def calculate_fitness(self, c):
        def prepare_data(x, y, c):
            y1 = np.expand_dims(y, axis=1).T
            y2 = y1.copy()
            for _ in range(self.population_size - 1):
                y2 = np.vstack((y2, y1))

            x1 = np.expand_dims(x, axis=1).T
            x2 = x1.copy()
            for _ in range(self.population_size - 1):
                x2 = np.vstack((x2, x1))

            c = c.T
            c1 = c.copy()
            for _ in range(len(x) - 1):
                c = np.dstack((c, c1))
            return x2, y2, c

        def F(x, c):
            return (c[0, :, :] * x + c[1, :, :]) / (x * x + c[2, :, :] * x + c[3, :, :])

        x, y = self.args
        x, y, c = prepare_data(x, y, c)
        D_ab = ((F(x, c) - y) ** 2).sum(axis=1)
        return D_ab

    # start the evolution
    def evolve(self, silence=False):
        nit, nfev = 0, 1

        for step in range(self.max_steps):
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            # Обновить скорость и вес
            self.v = self.w * self.v + self.c1 * r1 * (self.p - self.c) + self.c2 * r2 * (self.pg - self.c)
            self.c = self.v + self.c
            nit += 1
            nfev += 1
            fitness = self.calculate_fitness(self.c)

            # Лица, нуждающиеся в обновлении
            update_id = np.greater(self.individual_best_fitness, fitness)
            self.p[update_id] = self.c[update_id]
            self.individual_best_fitness[update_id] = fitness[update_id]

            # Новое поколение имеет меньшую физическую форму, поэтому обновите глобальную оптимальную форму и положение
            if np.min(fitness) < self.global_best_fitness:
                self.pg = self.c[np.argmin(fitness)]
                self.global_best_fitness = np.min(fitness)

            if(silence == False):
                print("nit:", nit)
                print('best fitness: %.5f, mean fitness: %.5f' % (self.global_best_fitness, np.mean(fitness)))
        info = {}
        info['nit'] = nit
        info['nfev'] = nfev
        return self.pg, info
