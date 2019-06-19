import multiprocessing as mp
import random
import string
from time import time
from datetime import timedelta

num_strings = 10

random.seed(123)

# Define an output queue
output = mp.Queue()

# define a example function
def rand_string(length, output):
    """ Generates a random string of numbers, lower- and uppercase chars. """
    rand_str = ''.join(random.choice(
        string.ascii_lowercase
        + string.ascii_uppercase
        + string.digits)
                       for i in range(length))
    output.put(rand_str)

# Setup a list of processes that we want to run
processes = [mp.Process(target=rand_string, args=(5, output)) for x in range(num_strings)]

t_start = time()

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
results = [output.get() for p in processes]

print(results)
print(f'Time to solve in parallel: {timedelta(seconds=time() - t_start)}')


# Serial execution
print(' ')
random.seed(123)


def rand_string2(length):
    """ Generates a random string of numbers, lower- and uppercase chars. """
    rand_str = ''.join(random.choice(
        string.ascii_lowercase
        + string.ascii_uppercase
        + string.digits)
                       for i in range(length))
    return rand_str

t_start = time()
output = []
for i in range(num_strings):
    output.append(rand_string2(5))

print(output)
print(f'Time to solve in series: {timedelta(seconds=time() - t_start)}')
