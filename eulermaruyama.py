import sys
import numpy as np
from math import sqrt
from sympy import *

x = symbols('x')
t = symbols('t')

def EulerMaruyama(f, g, t0, tfin, dt, x0):

    f = lambdify([t, x], f)
    g = lambdify([t, x], g)

    n = int(round((tfin-t0)/dt, 0))

    W = np.random.normal(size=n)*sqrt(dt)

    Y = [x0]

    for i in range(n):

        Y += [Y[i]
              + (f(t0 + i*dt, Y[i])
                 * dt)
              + (g(t0 + i*dt, Y[i])
                 * W[i])]

    return Y

def MilsteinMethod(f, g, t0, tfin, dt, x0):

    dgdx = lambdify([t, x], g.diff(x))
    f = lambdify([t, x], f)
    g = lambdify([t, x], g)

    n = int(round((tfin-t0)/dt, 0))

    W = np.random.normal(size=n)*sqrt(dt)

    Y = [x0]

    for i in range(n):

        Y += [Y[i]
              + (f(t0 + i*dt, Y[i])
                 * dt)
              + (g(t0 + i*dt, Y[i])
                 * W[i])
              + ((1/2)
                 * g(t0 + i*dt, Y[i])
                 * dgdx(t0 + i*dt, Y[i])
                 * ((W[i]**2) - dt))]

    return Y

if __name__ == "__main__":

    from sympy.parsing.sympy_parser import parse_expr
    from random import choice, randint
    import matplotlib.pyplot as plt

    METHODS = ['EULERMARUYAMA', 'MILSTEIN']

    METHOD = input('Choose the numerical method\n'
                   '(if illegal input is found, the method is chosen randomly): ').upper()

    T = input('\nNumber of simulations\n'
              '(if illegal input is found, the number is chosen randomly): ')

    print('\nFunctions must be in terms of t and x\n')

    f = input('f function: ')
    g = input('g function: ')

    print("\nWhat are the parameters?\n"
          "(Note that if illegal inputs are introduced,\n"
          "the default parameters are used instead:\n"
          "t0 = 0, tfin = 1, dt = (1/4), x0 = 1)")
    t0 = input('\nLower integration bound: ')
    tfin = input('\nUpper integration bound: ')
    dt = input('\nInterval step: ')
    x0 = input('\nInitial value: ')

    PARMS = [t0, tfin, dt, x0]

    try:
        PARMS = [float(i) for i in PARMS]

    except:
        print('\nDefault parameters are being used instead.')
        PARMS = [0, 1, 1/4, 1]

    if (f.upper() == 'TEST'
       or g.upper() == 'TEST'):

        f = '(1/3)*x**(1/3) + 6*x**(2/3)'
        g = 'x**(2/3)'

    try:
        f = parse_expr(f)
        g = parse_expr(g)

    except:
        print('Illegal functions found. Exiting script.')
        exit()

    try:
        T = int(T)
    except:
        T = randint(1, 20)

    if METHOD not in METHODS:

        METHOD = choice(METHODS)

    if METHOD == METHODS[1]:

        for i in range(T):

            plt.plot([i*2**(-2) for i in range(5)],
                     EulerMaruyama(f, g, *PARMS))

    else:

        for i in range(T):

            plt.plot([i*2**(-2) for i in range(5)],
                     MilsteinMethod(f, g, *PARMS))

    plt.title('Numerical integration by means of {}'.format(METHOD.capitalize()))
    plt.text(0, 10, 'f = {}\ng = {}'.format(f, g))
    plt.show()
