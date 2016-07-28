import random_scen

for i in range(1000):
  rewards, sinks = random_scen.random_scen(1, 585, [15, 15, 15])
  scen = {'rewards': rewards, 'sinks': sinks}
  random_scen.save3d(scen)

