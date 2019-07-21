from cvxopt import matrix, solvers
from cvxopt.base import spmatrix, spdiag
from cvxopt.modeling import dot, op, variable
import numpy as np

solvers.options['show_progress'] = False
solvers.options['msg_lev'] = 'GLP_MSG_OFF'
solvers.options['show_progress'] = False  # disable solver output
solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8
solvers.options['LPX_K_MSGLEV'] = 0  # previous versions


# https://github.com/adam5ny/blogs/blob/master/cvxopt/cvxopt_examples.py
def ce(Q, opQ, solver=None):
    na = Q.shape[0]
    nvars = na ** 2

    Q_flat = Q.flatten()
    opQ_flat = opQ.flatten()

    # Minimize matrix c (*=-1 to maximize)
    c = -np.array(Q_flat + opQ_flat, dtype="float")
    c = matrix(c)

    # Inequality constraints G*x <= h
    G = np.empty((0, nvars))

    # Player constraints
    for i in range(na):  # action row i
        for j in range(na):  # action row j
            if i == j: continue
            constraint = np.zeros(nvars)
            base_idx = i * na
            comp_idx = j * na
            for _ in range(na):
                constraint[base_idx + _] = Q_flat[comp_idx + _] - Q_flat[base_idx + _]
            G = np.vstack([G, constraint])

    # Opponent constraints
    Gopp = np.empty((0, nvars))
    for i in range(na):  # action row i
        for j in range(na):  # action row j
            if i == j: continue
            constraint = np.zeros(nvars)
            for _ in range(na):
                # constraint[base_idx + j * _] = opQ_flat[comp_idx + _] - opQ_flat[base_idx + _]
                constraint[i + _ * na] = opQ_flat[j + (_ * na)] - opQ_flat[i + (_ * na)]
            Gopp = np.vstack([Gopp, constraint])

    G = np.vstack([G, Gopp])
    G = np.matrix(G, dtype="float")
    G = np.vstack([G, -1. * np.eye(nvars)])
    h_size = len(G)
    G = matrix(G)
    h = np.array(np.zeros(h_size), dtype="float")
    h = matrix(h)

    # Equality constraints Ax = b
    A = np.matrix(np.ones(nvars), dtype="float")
    A = matrix(A)
    b = np.matrix(1, dtype="float")
    b = matrix(b)
    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)

    probs = np.array(sol['x'].T)[0]

    # Scale and normalize to prevent negative probabilities
    probs -= probs.min() + 0.
    return probs.reshape((na, na)) / probs.sum(0)


if __name__ == '__main__':
    q0 = np.random.rand(5, 5)
    q1 = np.random.rand(5, 5)
    ps = ce(q0, q1)
    print(ps)
    ps2 = np.sum(ps, axis=1)
    print(ps2)
    V = sum([ps2[a_] * q0[a_, o] for a_ in range(5)])
