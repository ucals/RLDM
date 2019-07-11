import numpy as np


class KWIKSolver(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.num_episodes = len(Y)
        self.hypothesis = np.full(len(X[0]), -1)
        self.memory = {'x': [], 'y': []}

    def update_hypothesis(self, x, y):
        if y == 1:
            for i in range(len(x)):
                if x[i] == 1:
                    self.hypothesis[i] = 1

        elif y == 0:
            pass

    def update_memory(self, x, y):
        self.memory['x'].append(x)
        self.memory['y'].append(y)

    def retrieve_from_memory(self, x):
        for i in range(len(self.memory['x'])):
            if x == self.memory['x'][i]:
                return self.memory['y'][i]

        return None

    def predict(self, x):
        # First check whether there is a prediction in memory
        prediction = self.retrieve_from_memory(x)
        if prediction is not None:
            print('from memory')
            return prediction

        # Get the hypothesis for the people present
        hyp_present = np.ma.masked_array(self.hypothesis,
                                         mask=1 - np.array(x)).compressed()

        # If there's no one, no fight
        if len(hyp_present) == 0:
            print('no one')
            return 0

        # If uncertainty
        if np.all(hyp_present == -1):
            print('uncertainty')
            return -1

        # If instigator might be present, and there's no chance of a peacemaker
        if np.any(hyp_present == 1) and not np.any(hyp_present == 0):
            print('instigator')
            return 1

        # If there's a chance of a peacemaker
        if np.any(hyp_present == 0):
            print('peacemaker')
            return 0

    def solve(self):
        result = [self.Y[0]]
        self.update_memory(self.X[0], self.Y[0])
        for i in range(1, self.num_episodes):
            self.update_hypothesis(self.X[i - 1], self.Y[i - 1])
            prediction = self.predict(self.X[i])
            result.append(prediction)
            if prediction != -1:
                self.update_memory(self.X[i], prediction)

            print(f'x: {self.X[i]}\npred: {prediction}\ny: {self.Y[i]}\nhypothesis: {self.hypothesis}\n')

        return result


if __name__ == '__main__':
    patrons = [[1, 1],
               [1, 0],
               [0, 1],
               [1, 1],
               [0, 0],
               [1, 0],
               [1, 1]]
    occurrences = [0, 1, 0, 0, 0, 1, 0]

    kwik = KWIKSolver(patrons, occurrences)
    solution = kwik.solve()
    print(solution)
    #assert solution == [0, -1, 0, 0, 0, 1, 0]
