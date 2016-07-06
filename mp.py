from mdp import ClassicMDP
from gridworld import Grid
from policy import Policy
from scikit_supervise import ScikitSupervise
import scenarios
from analysis import Analysis
import numpy as np
from multiprocessing import Process, Array
import ctypes
import multiprocessing as mp
import time


scen = scenarios.scenario0

grid = Grid(15, 15)
grid.reward_states = scen['rewards']
grid.sink_states = scen['sinks']
mdp = ClassicMDP(Policy(grid), grid)

an = Analysis(15,15, 1, scen['rewards'], scen['sinks'])

#mdp.value_iteration()
#mdp.save_policy('policies/mp.p')
mdp.load_policy('policies/mp.p')
value_iter_pi = mdp.pi

sup = ScikitSupervise(grid, mdp, value_iter_pi, classifier=None, moves=20)

SAMP = 40
reward = 0.0
states = []
for i in range(SAMP):
    sup.record=False
    mdp.pi = value_iter_pi
    sup.rollout()
    states = list(states) + list(sup.get_recent_rollout_states())
    reward += sup.get_reward() / float(SAMP)

""""
#an.count_states(np.array(states))
#an.show_states()
print "Avg. reward: " + str(reward)
T = 80000


a = np.zeros(10)
print "Original: " + str(a[:2])
start = time.clock()
for i in range(T * 2):
    a[0] += 1

end = time.clock()
print "Resulting: " + str(list(a)[:2])
print "Time elapsed" + str(end - start) + '\n'


a = np.zeros(10)
shared_arr = mp.Array(ctypes.c_double, a)
print "Original: " + str(a[:2])
def f():
    for i in range(T):
        with shared_arr.get_lock():
            arr = np.frombuffer(shared_arr.get_obj())
            arr[0] += 1


p = Process(target=f, args=())
p2 = Process(target=f, args=())
start = time.clock()
p.start()
p2.start()

p.join()
p2.join()
end = time.clock()
print "Resulting: " + str(list(shared_arr)[:2])
print "Time elapsed: " + str(end - start) + ' \n'



"""
