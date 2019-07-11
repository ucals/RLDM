from pulp import *
import numpy as np


def test_pulp():
    prob = LpProblem("The Whiskas Problem",LpMinimize)

    # The 2 variables Beef and Chicken are created with a lower limit of zero
    x1=LpVariable("ChickenPercent",0,None,LpInteger)
    x2=LpVariable("BeefPercent",0)

    # The objective function is added to 'prob' first
    prob += 0.013*x1 + 0.008*x2, "Total Cost of Ingredients per can"

    # The five constraints are entered
    prob += x1 + x2 == 100, "PercentagesSum"
    prob += 0.100*x1 + 0.200*x2 >= 8.0, "ProteinRequirement"
    prob += 0.080*x1 + 0.100*x2 >= 6.0, "FatRequirement"
    prob += 0.001*x1 + 0.005*x2 <= 2.0, "FibreRequirement"
    prob += 0.002*x1 + 0.005*x2 <= 0.4, "SaltRequirement"

    # The problem data is written to an .lp file
    prob.writeLP("WhiskasModel.lp")

    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])

    # Each of the variables is printed with it's resolved optimum value
    for v in prob.variables():
        print(v.name, "=", v.varValue)

    # The optimised objective function value is printed to the screen
    print("Total Cost of Ingredients per can = ", value(prob.objective))


def solve_rps(r):
    prob = LpProblem("Rock, Paper, Scissor", LpMaximize)

    v = LpVariable("TotalValue")
    p_r = LpVariable("RockProbability", 0, 1)
    p_p = LpVariable("PaperProbability", 0, 1)
    p_s = LpVariable("ScissorProbability", 0, 1)

    prob += v

    prob += p_r + p_p + p_s == 1
    prob += r[0][0] * p_r + r[1][0] * p_p + r[2][0] * p_s >= v
    prob += r[0][1] * p_r + r[1][1] * p_p + r[2][1] * p_s >= v
    prob += r[0][2] * p_r + r[1][2] * p_p + r[2][2] * p_s >= v

    prob.writeLP("rps.lp")
    prob.solve()
    print("Status:", LpStatus[prob.status])
    result = [0, 0, 0]
    for v in prob.variables():
        print(v.name, "=", v.varValue)
        if v.name == 'RockProbability':
            result[0] = v.varValue
        elif v.name == 'PaperProbability':
            result[1] = v.varValue
        elif v.name == 'ScissorProbability':
            result[2] = v.varValue

    return result


if __name__ == '__main__':
    matrix = np.array([[0.0, 3.79, -1.0], [-3.79, 0.0, 4.98], [1.0, -4.98, 0.0]])
    result = solve_rps(matrix)
    print(result)
