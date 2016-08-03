from policy import Action, Policy
from state import State
from mdp import ClassicMDP
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import IPython
import numpy as np
from random import randint
import scenarios
import itertools

class Grid():

    def __init__(self, *dims):
        self.dims = dims
        self.dim = len(dims)
        self.mdp = None
        self.time_steps = 0
        self.record_states = [] # list of states
        
        self._reward_states = self.get_reward_states()
        self._sink_states = self.get_sink_states()

        self._list_reward_states = None
        self._list_sink_states = None

        return


    def get_reward_states(self):
        """
            Fixed for now for at least 10x10 grid
        """
        return None

    def get_sink_states(self):
        """
            Fixed for now for at least 10x10 grid
        """
        return None

    @property
    def reward_states(self):
        if self._list_reward_states is None:
            self._list_reward_states = [State(*tup) for tup in self._reward_states.keys()]
        return self._list_reward_states
    
    @property
    def sink_states(self):
        if self._list_sink_states is None:
            self._list_sink_states = [State(*tup) for tup in self._sink_states.keys()]
        return self._list_sink_states

    def set_reward_states(self, rewards):
        self._reward_states = {}
        for state in rewards:
            self._reward_states[tuple(state.pos)] = 1


    def set_sink_states(self, sinks):
        self._sink_states = {}
        for state in sinks:
            self._sink_states[tuple(state.pos)] = 1


    def add_mdp(self, mdp):
        if self.mdp is not None:
            self.mdp.grid = None
        self.mdp = mdp
        self.mdp.grid = self
        self.mdp.state = State(*([0]*self.dim))
        self.record_states = [self.mdp.state]
        self.time_steps = 0

    def reset_mdp(self):
        self.add_mdp(self.mdp)

    def clear_record_states(self):
        self.record_states = [self.mdp.state]

    def get_dir(self, state, state_prime):
        dif = state_prime.subtract(state)
        action = state_prime.get_action(dif)
        return action

    def is_valid(self, state):
        results = [coord < dim and coord >=0 for coord, dim in zip(state.pos, self.dims)]
        return all(results)

    def get_state_prime(self, state, action):
        dif = state.difference(action)
        result = state.add(dif)
        return result

    def get_adjacent(self, state):
        adjacents = []
        for action in state.available_actions:
            state_prime = self.get_state_prime(state, action)
            if self.is_valid(state_prime):
                adjacents.append(state_prime)
        return adjacents

    def get_nearest_valid(self, state):
        if self.is_valid(state):
            return state
        new_pos = [0] * self.dim
        for i in range(self.dim):
            min_size = 0
            max_size = self.dims[i]
            new_pos[i] = max(0, state.pos[i])
            new_pos[i] = min(self.dims[i] - 1, start.pos[i])
        return State(*[new_pos])    

    def reward(self, state, action, state_prime):
        if not self.is_valid(state_prime):
            state_prime = state
        if Grid.contains_states(self._reward_states, state_prime):
            return 10
        elif Grid.contains_states(self._sink_states, state_prime):
            return -10
        else:
            #return -1.0
            return -.02

    """
    @staticmethod
    def contains_states(states, state):
        if state in states
        for s in states:
            if s.equals(state):
                return True
        return False
    """
    @staticmethod 
    def contains_states(states, state):
        try:
            states[tuple(state.pos)]
            return True
        except:
            return False

    def _draw_rewards(self, size):
        xs = []
        ys = []
        for state in self.reward_states:
            xs.append(state.pos[0])
            ys.append(state.pos[1])
        self.figure.scatter(xs, ys, s=size, c='g')         
        return
    
    def _draw_sinks(self, size):
        xs = []
        ys = []
        for state in self.sink_states:
            xs.append(state.pos[0])
            ys.append(state.pos[1])
        self.figure.scatter(xs, ys, s=size, c='r') 
        return

    
    def step(self,mdp):
        self.record_states.append(mdp.state)
        s, a = mdp.move()
        self.time_steps += 1
        return a
        
    def _animate(self, i):
        self.figure.autoscale(False)
        if i < len(self.record_states):
            width = self.dims[0]
            height = self.dims[1]
            xar = [self.record_states[i].pos[0]]
            yar = [self.record_states[i].pos[1]]
            robo_size = 15000 / self.dims[1] / self.dims[0]
            indicator_size = 30000 / self.dims[1] / self.dims[0]
            self.figure.clear()
            self._draw_rewards(indicator_size)
            self._draw_sinks(indicator_size)
            self.figure.scatter(xar,yar, s=robo_size)            
            self.figure.set_xlim([-.5, width - 1 +.5])
            self.figure.set_ylim([-.5, height - 1 + .5])                                

            width_range = np.arange(width)
            height_range = np.arange(height)

            plt.xticks(width_range)
            plt.yticks(height_range)
            
            plt.title("Step " + str(i))
            self.figure.set_yticks((height_range[:-1] + 0.5), minor=True)
            self.figure.set_xticks((width_range[:-1] + 0.5), minor=True)            
            self.figure.grid(which='minor', axis='both', linestyle='-')            
            
    def set_recording(self, recording):
        """
            recording is list of states
        """
        self.record_states = recording

    def show_recording(self):
        fig, self.figure = plt.subplots()
        print len(self.record_states)
        # All recordings should take ~10 seconds
        interval = float(7000) / float(len(self.record_states))
        try:
            an = animation.FuncAnimation(fig, self._animate, interval=interval, repeat=False)
            plt.show(block=False)
            plt.pause(interval * (len(self.record_states) + 1) / 1000)
            plt.close()
        except:
            return


    def get_all_states(self):
        ranges = [range(d) for d in self.dims]
        for tup in itertools.product(*ranges):
            yield State(*tup)




class HighVarInitStateGrid(Grid): 

    def add_mdp(self, mdp):
        if self.mdp is not None:
            self.mdp.grid = None
        self.mdp = mdp
        self.mdp.grid = self
        state = np.random.normal(0, 10, self.dim).astype(int)
        state = np.clip(state, 0, self.dims[0] - 1)
        self.mdp.state = State(*list(state))
        self.record_states = [self.mdp.state]
        self.time_steps = 0
        
class LowVarInitStateGrid(Grid):
    
    def add_mdp(self, mdp):
        if self.mdp is not None:
            self.mdp.grid = None
        self.mdp = mdp
        self.mdp.grid = self
        state = np.random.normal(0, 5, self.dim).astype(int)
        state = np.clip(state, 0, self.dims[0] - 1)
        self.mdp.state = State(*list(state))
        self.record_states = [self.mdp.state]
        self.time_steps = 0


"""
if __name__ == '__main__':
    grid = BasicGrid(15, 15)
    mdp = ClassicMDP(ClassicPolicy(grid), grid)
    
    #mdp.value_iteration()
    #mdp.save_policy()
    mdp.load_policy()
    
    #for i in range(40):
    #    grid.step(mdp)
    #grid.show_recording()
    
    dagger = SVMDagger(grid, mdp)
    dagger.rollout()            # rollout with supervisor policy
    grid.show_recording()
    #for _ in range(5):
    #    dagger.retrain()
    #    dagger.rollout()

"""

if __name__ == '__main__':
    
    grid = Grid(15, 15, 15, 15)
    scen = scenarios.scenario5
    grid.reward_states = scen['rewards']
    grid.sink_states = scen['sinks']
        
    mdp = ClassicMDP(Policy(grid), grid, state=State(0,0,0,0))
    #mdp.value_iteration()
    #mdp.save_policy(filename='4dim_policy.p')
    mdp.load_policy(filename='4dim_policy.p')

    for i in range(80):
        grid.step(mdp)
        print mdp.state
    

