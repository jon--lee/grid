import numpy as np

class Action():
    NONE = 0 
    def __init__(self):
        return

class Policy():
 
    def __init__(self, grid):
        self.arr = {}
        available_actions = range(grid.dim * 2 + 1)

    def get_next(self, state):
        if not tuple(state.pos) in self.arr:
            return 0.0
        return self.arr[tuple(state.pos)]
    
    def update(self, state, action):
        self.arr[tuple(state.pos)] = action

class SKPolicy(Policy):

    def __init__(self, est):
        self.est = est

    def get_next(self, state):
        return self.est.predict([list(state.pos)])
