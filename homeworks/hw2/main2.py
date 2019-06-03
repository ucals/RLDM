from scipy.optimize import fsolve,leastsq
import numpy as np


class TD_lambda:
    def __init__(self, probToState,valueEstimates,rewards):
        self.probToState = probToState
        self.valueEstimates = valueEstimates
        self.rewards = rewards
        self.td1 = self.get_vs0(1)

    def get_vs0(self, lambda_):
        probToState = self.probToState
        valueEstimates = self.valueEstimates
        rewards = self.rewards
        vs = dict(zip(['vs0','vs1','vs2','vs3','vs4','vs5','vs6'],list(valueEstimates)))

        vs5 = vs['vs5'] + 1 * (rewards[6] + 1 * vs['vs6'] - vs['vs5'])

        vs4 = vs['vs4'] + 1 * (rewards[5] + lambda_ * rewards[6] + lambda_ * vs['vs6'] + (1 - lambda_) * vs['vs5'] - vs['vs4'])

        vs3 = vs['vs3'] + 1 * (rewards[4] + lambda_ * rewards[5] + lambda_**2 * rewards[6] + lambda_**2 * vs['vs6'] + lambda_ * (1 - lambda_) * vs['vs5'] + (1 - lambda_) * vs['vs4'] - vs['vs3'])

        vs1 = vs['vs1'] + 1*(rewards[2]+lambda_*rewards[4]+lambda_**2*rewards[5]+lambda_**3*rewards[6]+lambda_**3*vs['vs6']+lambda_**2*(1-lambda_)*vs['vs5']+lambda_*(1-lambda_)*vs['vs4']+ \
                             (1-lambda_)*vs['vs3']-vs['vs1'])

        vs2 = vs['vs2'] + 1*(rewards[3]+lambda_*rewards[4]+lambda_**2*rewards[5]+lambda_**3*rewards[6]+lambda_**3*vs['vs6']+lambda_**2*(1-lambda_)*vs['vs5']+lambda_*(1-lambda_)*vs['vs4']+ \
                             (1-lambda_)*vs['vs3']-vs['vs2'])

        vs0 = vs['vs0'] + probToState*(rewards[0]+lambda_*rewards[2]+lambda_**2*rewards[4]+lambda_**3*rewards[5]+lambda_**4*rewards[6]+lambda_**4*vs['vs6']+lambda_**3*(1-lambda_)*vs['vs5']+ \
                                       +lambda_**2*(1-lambda_)*vs['vs4']+lambda_*(1-lambda_)*vs['vs3']+(1-lambda_)*vs['vs1']-vs['vs0']) + \
              (1-probToState)*(rewards[1]+lambda_*rewards[3]+lambda_**2*rewards[4]+lambda_**3*rewards[5]+lambda_**4*rewards[6]+lambda_**4*vs['vs6']+lambda_**3*(1-lambda_)*vs['vs5']+ \
                               +lambda_**2*(1-lambda_)*vs['vs4']+lambda_*(1-lambda_)*vs['vs3']+(1-lambda_)*vs['vs2']-vs['vs0'])
        return vs0

    def get_lambda(self,x0=0.5): #np.linspace(0.1,1,10)):
        return fsolve(lambda lambda_:self.get_vs0(lambda_)-self.td1, x0)


if __name__ == '__main__':
    #probToState = 0.5
    #valueEstimates = [0, 3, 8, 2, 1, 2, 0]
    #rewards = [0, 0, 0, 4, 1, 1, 1]
    probToState=0.62
    valueEstimates=[0.0,0,2.6,19.5,4.8,0.6,0.0]
    rewards=[-2.8,4.4,-1.1,1.1,-2.2,2.7,1.6]

    TD = TD_lambda(probToState,valueEstimates,rewards)
    print(TD.get_lambda())
    print(TD.td1)

