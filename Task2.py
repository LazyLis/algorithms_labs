import math
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize
import numpy as np

def task2_1():

    # f(x) = x**3
    def func_1(x):
        return x**3

    # f(x) = |x - 0.2|
    def func_2(x):
        return math.fabs(x - 0.2)

    # f(x) = xsin 1/x
    def func_3(x):
        return x * math.sin(1/x)

    # exhausitiv search
    def exhausitiv_search():


        def exhausitiv_search_m(a, b, f, name):
            min_y = 9999
            min_x = -1
            flag = 1
            step = 0.001
            while a <= b:
                temp = (f(a))
                if temp < min_y:
                    min_y = temp
                    min_x = a
                a += step
                flag += 1
            print(f'Minimum point in the {name} function by the exhausitiv search:\n'
                  f'x min:                           {round(min_x, 5)}\n'
                  f'iterations:                      {flag}\n'
                  f'number of function calculations: {flag}\n')


        exhausitiv_search_m(0, 1, func_1, 'first')
        exhausitiv_search_m(0, 1, func_2, 'second')
        exhausitiv_search_m(0.1, 1, func_3, 'third')

    def dychotomy():
        eps = 0.001

        def dychtomy_m(a, b, f, name):
            flag = 0
            while math.fabs(b - a) > eps:
                s = eps / 2 # the offset value s must be less than 2 times epsilon
                mid = (a+b) / 2
                x1 = mid - s
                x2 = mid + s
                if f(x1) < f(x2):
                    b = x1
                else:
                    a = x2
                flag +=1
            print(f'Minimum point in the {name} function by the dichotomy method:\n'
                  f'x min:                           {round((a + b) / 2, 5)}\n'
                  f'number of iterations:            {flag}\n'
                  f'number of function calculations: {flag * 2}\n')


        dychtomy_m(0, 1, func_1, 'first')
        dychtomy_m(0, 1, func_2, 'second')
        dychtomy_m(0.1, 1, func_3, 'third')
    # golden section method

    def golden_section():
        eps = 0.001

        def golden_section_m(a, b, f, name):
            flag = 1
            x1 = a + ((3 - 5 ** 0.5) / 2) * (b - a)
            # x1 = a + (2 / (3 + 5 ** 0.5)) * (b - a)
            x2 = b + ((5 ** 0.5 - 3) / 2) * (b - a)
            # x2 = a + (2 / (1 + 5 ** 0.5)) * (b - a)
            y1 = f(x1)
            y2 = f(x2)
            while math.fabs(a - b) > eps:
                if y1 <= y2:
                    b = x2
                    x2 = x1
                    y2 = y1
                    x1 = a + ((3 - 5 ** 0.5) / 2) * (b - a)
                    # x1 = a + (2 / (3 + 5 ** 0.5)) * (b - a)
                    y1 = f(x1)
                else:
                    a = x1
                    x1 = x2
                    y1 = y2
                    x2 = b + ((5 ** 0.5 - 3) / 2) * (b - a)
                    # x2 = a + (2 / (1 + 5 ** 0.5)) * (b - a)
                    y2 = f(x2)

                flag += 1

            print(f'Minimum point in the {name} function by the golden section method:\n'
                  f'x min:                           {round((a + b) / 2, 5)}\n'
                  f'number of iterations:            {flag}\n'
                  f'number of function calculations: {flag + 1}\n')

        golden_section_m(0, 1, func_1, 'first')
        golden_section_m(0, 1, func_2, 'second')
        golden_section_m(0.1, 1, func_3, 'third')


    exhausitiv_search()
    dychotomy()
    golden_section()


def task2_2():
    def D_ab_nonli(c):

        def F_ab_nonlinear(x, c):
            return c[0] / (1 + c[1] * x)

        random.seed(1)
        alpha = random.random()
        betta = random.random()

        np.random.seed(1)
        x = np.array([k / 100 for k in range(101)])
        y = np.array(alpha * x + betta + np.random.normal(0., 1., size=len(x)))

        D_ab = sum((F_ab_nonlinear(x, c) - y) ** 2)
        return D_ab

    def D_ab(c):

        def F_ab_linear(x, c):
            return c[0] * x + c[1]

        random.seed(1)
        alpha = random.random()
        betta = random.random()

        np.random.seed(1)
        x = np.array([k / 100 for k in range(101)])
        y = np.array(alpha * x + betta + np.random.normal(0., 1., size=len(x)))

        D_ab = sum((F_ab_linear(x, c) - y) ** 2)
        return D_ab

    def exha_s_m_ab(f, name):
        a = -0.499
        b = -0.499
        min_y = 9999
        min_a = -1
        min_b = -1
        flag = 0
        step = 0.001
        for i in range(2001):
            b = -0.499
            for i in range(2001):
                temp = f(np.array([a, b]))
                if temp < min_y:
                    min_y = temp
                    min_a = a
                    min_b = b
                b += step
                flag += 1
            a += step

        print(f"exhaustive search, {name}\n"
              f"x: {[min_a, min_b]}\n"
              f"f(x) = {min_y}\n"
              f"number of iterations: {flag}\n"
              f"number of function calculations: {flag}\n")
        return [min_a, min_b]

    def gauss(name):


        def D_ab1(c, x, y, b):
            def F_ab_linear():
                return c * x + b

            return sum((F_ab_linear() - y) ** 2)

        def D_ab2(b, x, y, c):
            def F_ab_linear():
                return c * x + b

            return sum((F_ab_linear() - y) ** 2)

        def D_ab_nonli1(c, x, y, b):
            def F_ab_nonlinear():
                return c / (1 + b * x)

            return sum((F_ab_nonlinear() - y) ** 2)

        def D_ab_nonli2(b, x, y, c):
            def F_ab_nonlinear():
                return c / (1 + b * x)

            return sum((F_ab_nonlinear() - y) ** 2)

        if name == 'linear approximant':
            f_1 = D_ab1
            f_2 = D_ab2
        else:
            f_1 = D_ab_nonli1
            f_2 = D_ab_nonli2


        eps = 0.001
        random.seed(1)
        alpha = random.random()
        betta = random.random()

        np.random.seed(1)
        x = np.array([k / 100 for k in range(101)])
        y = np.array(alpha * x + betta + np.random.normal(0., 1., size=len(x)))

        a, b = 0.8, 0.8
        enter_point, last_call = 1, (a, b)
        last_func, enter_func = -1000, 1
        iter_number, func_number, loop_num = 0, 0, 1

        while enter_point >= eps:

            if loop_num % 2 == 0:
                res = minimize(f_1, np.array([a]), args=(x, y, b), method='nelder-mead',
                               options={"xatol": 1e-3})
                a = res.get("x")[0]
            else:
                res = minimize(f_2, np.array([b]), args=(x, y, a), method='nelder-mead',
                               options={"xatol": 1e-3})
                b = res.get("x")[0]

            point = (a, b)

            iter_number += 1 + res.get('nit')
            func_number += res.get('nfev')
            loop_num += 1

            enter_point = abs((point[0] - last_call[0]) + (point[1] - last_call[1]))
            enter_func = res.get("fun") - last_func
            last_call, last_func = point, res.get("fun")

        # print("Точка:", a, b)
        # print(res.get("fun"))

        print(f"Gauss method, {name}\n"
              f"x: {[a, b]}\n"
              f"f(x) = {res.get('fun')}\n"
              f"number of iterations: {iter_number}\n"
              f"number of function calculations: {func_number}\n")
        return [a, b]

    def nelder_method(name, f):

        ab_0 = np.array([0.5, 0.5])
        res1 = minimize(f, ab_0, method='nelder-mead',
                        options={'disp': False, "xatol": 1e-3})
        print('\n')
        print(f"Nelder-Mead, {name}\n"
              f"x: {res1['x']}\n"
              f"f(x) = {res1['fun']}\n"
              f"number of iterations: {res1['nit']}\n"
              f"number of function calculations: {res1['nfev']}\n")

        return res1['x']

    random.seed(1)
    alpha = random.random()
    betta = random.random()

    np.random.seed(1)
    x = np.array([k / 100 for k in range(101)])
    y = np.array(alpha * x + betta + np.random.normal(0., 1., size=len(x)))

    x_line = np.array([k / 100 for k in range(101)])
    y_line = np.array(alpha * x + betta)

    def visual(name, exh_s, nelder_me, gauss):
        if name == 'linear approximant':
            y_line_exh = np.array(exh_s[0] * x + exh_s[1])
            y_line_gauss = np.array(gauss[0] * x + gauss[1])
            y_line_nelder_m = np.array(nelder_me[0] * x + nelder_me[1])
        else:
            y_line_exh = np.array(exh_s[0] / (1 + exh_s[1] * x))
            y_line_gauss = np.array(gauss[0] / (1 + gauss[1] * x))
            y_line_nelder_m = np.array(nelder_me[0] / (1 + nelder_me[1] * x))

        fig, ax = plt.subplots()
        plt.title(name)
        ax.plot(x_line, y_line, label='generating line', color='b')
        ax.plot(x, y, 'ro', label='Generated data')
        ax.plot(x_line, y_line_exh, label='exhaustive search', color='g')
        ax.plot(x_line, y_line_nelder_m, label='Nelder-Mead method', color='m')
        ax.plot(x_line, y_line_gauss, label='Gauss method.', color='y')
        plt.grid()
        ax.legend()
        plt.show()


    name_f1 = 'linear approximant'
    name_f2 = 'rational approximant'

    visual(name_f1,
           exha_s_m_ab(D_ab, name_f1),
           nelder_method(name_f1, D_ab),
           gauss(name_f1))

    visual(name_f2,
           exha_s_m_ab(D_ab_nonli, name_f2),
           nelder_method(name_f2, D_ab_nonli),
           gauss(name_f2))


task2_1()
task2_2()
