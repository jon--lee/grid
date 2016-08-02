import random_scen

for i in range(10):
    rewards, sinks = random_scen.random_scen(1, 3, [15, 15, 15])
    scen = {'rewards': rewards, 'sinks': sinks}
    #random_scen.save2d(scen)
    random_scen.saveSimple(scen)

#for i in range(100):
#  rewards, sinks = random_scen.random_scen(1, 585, [15, 15, 15])
#  scen = {'rewards': rewards, 'sinks': sinks}
#  random_scen.save3d(scen)

