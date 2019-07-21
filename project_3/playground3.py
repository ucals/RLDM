import numpy as np
from cvxopt import matrix, solvers
import pulp


def ce(A, solver=None):
    num_vars = len(A)
    # maximize matrix c
    c = [sum(i) for i in A] # sum of payoffs for both players
    c = np.array(c, dtype="float")
    c = matrix(c)
    c *= -1 # cvxopt minimizes so *-1 to maximize
    # constraints G*x <= h
    G = build_ce_constraints(A=A)
    print(G)
    G = np.vstack([G, np.eye(num_vars) * -1]) # > 0 constraint for all vars
    h_size = len(G)
    G = matrix(G)
    h = [0 for i in range(h_size)]
    h = np.array(h, dtype="float")
    h = matrix(h)
    # contraints Ax = b
    A = [1 for i in range(num_vars)]
    A = np.matrix(A, dtype="float")
    A = matrix(A)
    b = np.matrix(1, dtype="float")
    b = matrix(b)
    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)
    return sol


def build_ce_constraints(A):
    num_vars = int(len(A) ** (1/2))
    G = []
    # row player
    for i in range(num_vars): # action row i
        for j in range(num_vars): # action row j
            if i != j:
                constraints = [0 for i in A]
                base_idx = i * num_vars
                comp_idx = j * num_vars
                for k in range(num_vars):
                    constraints[base_idx+k] = (- A[base_idx+k][0]
                                               + A[comp_idx+k][0])
                G += [constraints]
    # col player
    for i in range(num_vars): # action column i
        for j in range(num_vars): # action column j
            if i != j:
                constraints = [0 for i in A]
                for k in range(num_vars):
                    constraints[i + (k * num_vars)] = (
                            - A[i + (k * num_vars)][1]
                            + A[j + (k * num_vars)][1])
                G += [constraints]
    return np.matrix(G, dtype="float")


def pulp_ce(A):
    vars = pulp.LpVariable.dicts("Util", range(4), 0, 1.0)

    # Create the 'prob' variable to contain the problem data
    prob = pulp.LpProblem("Chicken problem", pulp.LpMaximize)
    sum_rewards = [sum(i) for i in A]

    prob += pulp.lpSum([vars[i] * sum_rewards[i] for i in range(len(A))])
    prob += pulp.lpSum([vars[i] for i in range(len(A))]) == 1

    G = build_ce_constraints(A)
    for r in range(G.shape[0]):
        prob += pulp.lpSum([G[r, c] * vars[c] for c in range(G.shape[1])]) <= 0

    prob.solve()
    probabilities = [vars[i].varValue for i in range(len(A))]
    print(probabilities)
    probabilities = np.array(probabilities).reshape(2, 2)
    print(probabilities)


if __name__ == '__main__':
    x = np.random.randint(4, size=(4, 4))
    print(x)
    print(np.sum(x, axis=1))


    exit(0)
    A = [[6, 6], [2, 7], [7, 2], [0, 0]]
    pulp_ce(A)

    exit(0)

    sol = ce(A=A, solver=None)
    probs = sol["x"]
    print(probs)
