import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize
from scipy.optimize import least_squares


def E(y, a, b, name):
    if name == 'linear':
        ff = np.array([a * (x / 100) + b for x in range(101)])
    else:
        ff = np.array([a / (1 + b * (x / 100)) for x in range(101)])
    return np.dot((y-ff).T, (y-ff))

# Gradient descent method
def gradient_descent(name, y):

    # partial derivatives
    def dE_dab(y, a, b, name):
        if name=='linear':
            return gradient_lin((a, b), x, y)
        else:
            return gradient_nonlin((a, b), x, y)

    aa = 0.4
    bb = 0.4
    l1 = 0.0001
    l2 = 0.0005
    # N = 50

    delta = 100
    nit, nfev = 0, 0
    while delta >= 0.001:
        nit += 1
        previous_c = [aa, bb]
        nfev += 2
        da, db = dE_dab(y, aa, bb, name)
        aa = aa - l1 * da
        bb = bb - l2 * db
        delta = abs(aa - previous_c[0]) + abs(bb - previous_c[1])

    print('\n')
    print(f"Gradient descent:\n"
          f"x: {(aa, bb)}\n"
          f"f(x) = {E(y, aa, bb, name)}\n"
          f"number of iterations: {nit}\n"
          f"Function evaluations: {nfev}\n")
    return [aa, bb]


# Numerical computation of hessian
def hessian_num(c, x, y, func, eps=1e-6):
    a_arg, b_arg = c[0], c[1]

    func_ab = func([a_arg, b_arg], x, y)
    func_a_eps_b = func([a_arg + eps, b_arg], x, y)
    func_a_b_eps = func([a_arg, b_arg + eps], x, y)

    part_deriv_aa = (func_a_eps_b - 2 * func_ab + func([a_arg - eps, b_arg], x, y)) / (eps ** 2)
    part_deriv_bb = (func_a_b_eps - 2 * func_ab + func([a_arg, b_arg - eps], x, y)) / (eps ** 2)
    part_deriv_ab = (func([a_arg + eps, b_arg + eps], x, y) - func_a_eps_b - func_a_b_eps + func_ab) / (eps ** 2)

    return np.array([[part_deriv_aa, part_deriv_ab], [part_deriv_ab, part_deriv_bb]])

# Objective function, linear approximant
def D_ab_lin(c, x, y):
    func = c[0]*x + c[1] - y
    return np.dot(func, func)

# Objective function, rational approximant
def D_ab_nonlin(c, x, y):
    func = c[0] / (1 + c[1]*x) - y
    return np.dot(func, func)

# Gradient function, linear approximant
def gradient_lin(c, x, y):
    da = 2*np.dot(c[0]*x + c[1] - y, x)
    db = 2*(c[0] * x + c[1] - y).sum()
    return np.array([da, db])

# Gradient function, rational approximant
def gradient_nonlin(c, x, y):
    da = 2*c[0] * ((1/(c[1]*x + 1))**2).sum() - 2 * (y / (c[1] * x + 1)).sum()
    db = -2*c[0]*c[0] * (x / (c[1]*x+1)**3).sum() + 2*c[0] * (y*x / (c[1] * x + 1)**2).sum()
    return np.array([da, db])

# Newton's method
def newton(c, x, y, D_ab, gradient, hesse, func_type, eps=0.001):
    alpha = 0.01
    delta = 1
    nit, neval = 0, 0
    previous_d = D_ab(c, x, y)

    while delta >= eps:
        nit += 1
        p = np.linalg.inv(hesse(c, x, y, D_ab)) @ gradient(c, x, y)
        neval += 2 if func_type == 'linear' else 4
        c = c - alpha * p
        current_d = D_ab(c, x, y)
        neval += (6 + 1)
        delta = np.fabs(current_d - previous_d)
        previous_d = current_d

    print('\n')
    print(f"Newton:\n"
          f"x: {c}\n"
          f"f(x) = {D_ab(c, x, y)}\n"
          f"number of iterations: {nit}\n"
          f"Function evaluations: {neval}\n")
    return c

# Levenberg-Marquardt method
def levenberg_marquardt(x, y, c0, type):

    def rational(c, x, y):
        return c[0] / (1 + c[1]*x) - y
    def linear(c, x, y):
        return c[0] * x + c[1] - y

    func = rational if type == 'rational' else linear

    # method='lm' requires object function that is not a scalar (it has it dot product inside the least_squares method)
    res = least_squares(func, c0, method='lm', xtol=1e-3, args=(x, y), verbose=1)
    print('\n')
    print(f"Levenberg-Marquardt:\n"
          f"x: {res.x}\n"
          f"f(x) = {np.dot(func(res.x, x, y), func(res.x, x, y))}\n"
          # f"number of iterations: {res['nit']}\n"
          f"Function evaluations: {res.nfev}\n")
    return res.x


random.seed(1)
at = random.random()
bt = random.random()

np.random.seed(1)
x = np.array([k / 100 for k in range(101)])
y = np.array(at * x + bt + np.random.normal(0., 1., size=len(x)))
method_type = 'linear'

print('\nConjugate  gradient:')
if method_type == 'linear':

    cg = minimize(D_ab_lin, np.array([1, 1]), method='CG', args=(x, y),  jac=gradient_lin, \
                       options={'eps': 1e-3, 'disp': True})
    newt = newton(np.array([0.59, 0.6]), x, y, D_ab_lin, gradient_lin, hessian_num, method_type)
    lm = levenberg_marquardt(x, y, np.array([1, 1]), method_type)
    gd = gradient_descent(method_type, y)
else:
    cg = minimize(D_ab_nonlin, np.array([0.450, 0.450]), method='CG', args=(x, y),  jac=gradient_nonlin, \
                        options={'disp': True, 'eps': 1e-3})
    newt = newton(np.array([0.59, 0.6]), x, y, D_ab_nonlin, gradient_nonlin, hessian_num, method_type)
    lm = levenberg_marquardt(x, y, np.array([1, 1]), method_type)
    gd = gradient_descent(method_type, y)

fig, ax = plt.subplots()
plt.title(method_type + ' approximant')
# ax.plot(x, (at * x + bt), label='generating line', color='yellow')
ax.plot(x, (newt[0] * x + newt[1]), label='Newton', color='magenta')
ax.plot(x, (cg.x[0] * x + cg.x[1]), label='Conjugate gradient', color='orange')
ax.plot(x, (lm[0] * x + lm[1]), label='Levenberg-Marquardt', color='red')
ax.plot(x, (gd[0] * x + gd[1]), label='Gradient descent', color='blue')
ax.plot(x, y, 'ro', markersize=2, label='Generated data')

plt.grid()
ax.legend()
plt.show()
