import numpy as np
import random
class Action():
    NONE = 0 
    def __init__(self):
        return

class Policy():
 
    def __init__(self, grid):
        self.arr = {}
        available_actions = range(grid.dim * 2 + 1)

    def get_next(self, state):
        #if not tuple(state.pos) in self.arr:
        #    return 0.0
        return self.arr[tuple(state.pos)]
    
    def update(self, state, action):
        self.arr[tuple(state.pos)] = action

    def get_actual_next(self, state):
        return self.get_next(state)

class SKPolicy(Policy):

    def __init__(self, est):
        self.est = est

    def get_next(self, state):
        return self.est.predict([list(state.pos)])

    def get_actual_next(self, state):
        return self.est.predict([list(state.pos)])


"""class SKNoisy5(Policy):
    EPS = .5

    def __init__(self, est):
        self.est = est

    def get_next(self, state):
        if random.random() > self.EPS:
            return self.est.predict([list(state.pos)])
        else:
            return random.choice(state.available_actions)

    def get_actual_next(self, state):
        return self.est.predict([list(state.pos)])
   """ 

"""class NoisyPolicy(Policy):

    EPS = 0.3
    #EPS = .9

    def get_next(self, state):
        if random.random() > NoisyPolicy.EPS:
            return Policy.get_next(self, state)
        else:
            return random.choice(state.available_actions)

    def get_actual_next(self, state):
        return Policy.get_next(self, state)
"""
 

class NoisyPolicy1(Policy):

    EPS = 0.1

    def get_next(self, state):
        if random.random() > self.EPS:
            return Policy.get_next(self, state)
        else:
            return random.choice(state.available_actions)

    def get_actual_next(self, state):
        return Policy.get_next(self, state)

class NoisyPolicy2(Policy):

    EPS = 0.2

    def get_next(self, state):
        if random.random() > self.EPS:
            return Policy.get_next(self, state)
        else:
            return random.choice(state.available_actions)

    def get_actual_next(self, state):
        return Policy.get_next(self, state)



class NoisyPolicy3(Policy):

    EPS = 0.3
    #EPS = .9

    def get_next(self, state):
        if random.random() > self.EPS:
            return Policy.get_next(self, state)
        else:
            return random.choice(state.available_actions)

    def get_actual_next(self, state):
        return Policy.get_next(self, state)

class NoisyPolicy4(Policy):

    EPS = 0.4

    def get_next(self, state):
        if random.random() > self.EPS:
            return Policy.get_next(self, state)
        else:
            return random.choice(state.available_actions)

    def get_actual_next(self, state):
        return Policy.get_next(self, state)



class NoisyPolicy6(Policy):

    EPS = 0.6
    #EPS = .9

    def get_next(self, state):
        if random.random() > self.EPS:
            return Policy.get_next(self, state)
        else:
            return random.choice(state.available_actions)

    def get_actual_next(self, state):
        return Policy.get_next(self, state)

class NoisyPolicy7(Policy):

    EPS = 0.7

    def get_next(self, state):
        if random.random() > self.EPS:
            return Policy.get_next(self, state)
        else:
            return random.choice(state.available_actions)

    def get_actual_next(self, state):
        return Policy.get_next(self, state)


class NoisyPolicy8(Policy):

    EPS = 0.8

    def get_next(self, state):
        if random.random() > self.EPS:
            return Policy.get_next(self, state)
        else:
            return random.choice(state.available_actions)

    def get_actual_next(self, state):
        return Policy.get_next(self, state)


class NoisyPolicy9(Policy):

    EPS = 0.9
    #EPS = .9

    def get_next(self, state):
        if random.random() > self.EPS:
            return Policy.get_next(self, state)
        else:
            return random.choice(state.available_actions)

    def get_actual_next(self, state):
        return Policy.get_next(self, state)

class NoisyPolicy99(Policy):

    EPS = 0.99
    #EPS = .9

    def get_next(self, state):
        if random.random() > self.EPS:
            return Policy.get_next(self, state)
        else:
            return random.choice(state.available_actions)

    def get_actual_next(self, state):
        return Policy.get_next(self, state)


class NoisyPolicy5(Policy):

    EPS = 0.5
    #EPS = .9

    def get_next(self, state):
        if random.random() > self.EPS:
            return Policy.get_next(self, state)
        else:
            return random.choice(state.available_actions)

    def get_actual_next(self, state):
        return Policy.get_next(self, state)

class NoisyPolicy0(Policy):

    EPS = 0.0
    #EPS = .9

    def get_next(self, state):
        if random.random() > self.EPS:
            return Policy.get_next(self, state)
        else:
            raise Exception("eps = 0 policy choose random action... that should not happen")
            return random.choice(state.available_actions)

    def get_actual_next(self, state):
        return Policy.get_next(self, state)


class NoisyPolicy10(Policy):

    EPS = 1.0
    #EPS = .9

    def get_next(self, state):
        if random.random() > self.EPS:
            return Policy.get_next(self, state)
        else:
            return random.choice(state.available_actions)

    def get_actual_next(self, state):
        return Policy.get_next(self, state)


