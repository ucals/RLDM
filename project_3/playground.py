import numpy as np
from collections import defaultdict
import random
from cvxopt import matrix, solvers
import pulp


def choice_experiments():
    x = np.random.randint(2, size=(5, 5))
    print(x)
    winner = np.argwhere(x == np.amax(x))
    print(winner)
    choice = np.random.choice(winner.shape[0])
    print(winner[choice])


def remove_negatives():
    x = [0, 1, 2, -1, 3, -2, 0]
    print(x)

    y = [a if a >= 0 else 0 for a in x]
    print(y)


def argmax():
    Q1 = defaultdict(lambda: np.zeros(5))
    print(Q1[0])
    Q1[0] = np.random.rand(5)
    print(Q1[0])
    print(np.argmax(Q1[0]))


def adding_rows():
    np.random.seed(1)
    random.seed(1)
    x = np.random.randint(4, size=(5, 5))
    print(x)
    print(np.sum(x, axis=1))


def loop_i_j():
    for i in range(5):
        for j in range(5):
            if (i == j) or (j < i):
                continue

            print(f'{i}-{j}')


def experiment():
    np.random.seed(1)
    random.seed(1)
    x = np.random.randint(4, size=(5, 5))
    print(x)
    probs = np.random.rand(5)
    print(probs)
    print(probs * x)
    print(np.sum(probs * x, axis=1))
    print(np.max(np.sum(probs * x, axis=1)))


def lp1():
    c = matrix([-1., -1.])
    G = matrix([[4., 2., -5.], [-1., 1., 2.]])
    h = matrix([8., 10., 2.])
    solution = solvers.lp(c, G, h)
    print(solution['x'])


if __name__ == '__main__':
    c = matrix([1., 1., 1., 1.])
    G = matrix([[1., 0., 1., 0.], [-2., 0., 0., -1.],
                [0., -1., -2., 0.], [0., 2., 0., 2.]])
    h = matrix([0., 0., 0., 0.])
    A = matrix([[1.], [1.], [1.], [1.]])
    b = matrix(1.)
    solution = solvers.lp(c, G, h, A, b, solver='glpk')
    print(solution['x'])

    # Pulp
    prob = pulp.LpProblem("uCE-Q", pulp.LpMaximize)
    p_1 = pulp.LpVariable("Action_0_Probability_Put", 0, 1, pulp.LpContinuous)
    p_2 = pulp.LpVariable("Action_1_Probability_North", 0, 1, pulp.LpContinuous)
    p_3 = pulp.LpVariable("Action_2_Probability_East", 0, 1, pulp.LpContinuous)
    p_4 = pulp.LpVariable("Action_3_Probability_South", 0, 1, pulp.LpContinuous)

    prob += 12 * p_1 + 9 * p_2 + 9 * p_3
    prob += p_1 + p_2 + p_3 + p_4 == 1
    prob += - p_1 + 2 * p_2 >= 0.
    prob += p_3 - 2 * p_4 >= 0.
    prob += - p_1 + 2 * p_3 >= 0.
    prob += p_2 - 2 * p_4 >= 0.

    #prob.solve()
    options = {} #{'LPX_K_MSGLEV': 0, 'msg_lev': "GLP_MSG_OFF"}
    prob.solve(pulp.solvers.GLPK(msg=0, keepFiles=1, options=options)) #(pulp.solvers.GLPK(msg = 1))

    probabilities = [p_1.varValue, p_2.varValue, p_3.varValue, p_4.varValue]
    print(probabilities)



