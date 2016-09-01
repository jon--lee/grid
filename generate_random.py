import random_scen

for i in range(100):
    rewards, sinks = random_scen.random_scen(1, 180, [15, 15])
    scen = {'rewards': rewards, 'sinks': sinks}
    random_scen.save_dense2d(scen)

#sparse
"""
for i in range(200):
    rewards, sinks = random_scen.random_scen(1, 200, [15, 15, 15])
    scen = {'rewards': rewards, 'sinks': sinks}
    #random_scen.saveSimple(scen)
    random_scen.save_sparse3d(scen)
"""
#for i in range(100):
#  rewards, sinks = random_scen.random_scen(1, 585, [15, 15, 15])
#  scen = {'rewards': rewards, 'sinks': sinks}
#  random_scen.save3d(scen)

