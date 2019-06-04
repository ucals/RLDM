import numpy as np
import json

if __name__ == '__main__':
    d = dict()
    d['gamma'] = 0.75
    d['states'] = list()
    d['states'].append('a')
    d['states'].append('b')
    d['states'].append('c')
    print(d)

    j = json.dumps(d, indent=4)
    print(j)

