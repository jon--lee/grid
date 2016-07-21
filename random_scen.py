import random
from state import State
import numpy as np
import os
import pickle

def random_scen(n_rewards, n_sinks, dims):
    seen = {}
    reward_states = find(n_rewards, seen, dims)
    sink_states = find(n_sinks, seen, dims)
    return reward_states, sink_states

def find(n, seen, dims):
    states = []
    for i in range(n):
        pos = [0] * len(dims)
        for i in range(len(dims)):
            pos[i] = random.randint(0, dims[i]-1)
        while tuple(pos) in seen:
            for i in range(len(dims)):
                pos[i] = random.randint(0, dims[i]-1)
        seen[tuple(pos)] = True
        states.append(State(*pos))
    return states


def save3d(scen):
    i = 0
    path = 'scenarios3d/scen' + str(i) + '.p'
    while os.path.exists(path):
        i += 1
        path = 'scenarios3d/scen' + str(i) + '.p'
    with open(path, 'w') as f:
        pickle.dump(scen, f)
    return path

def load(path):
    with open(path, 'r') as f:
        return pickle.load(f)

if __name__ == '__main__':
    rewards, sinks = random_scen(1, 1000, [15, 15, 15])
    scen = {'rewards': rewards,
            'sinks': sinks}
    save3d(scen)

