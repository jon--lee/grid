from gridworld import Grid
from mdp import ClassicMDP
import scenarios
from policy import Policy

grid = Grid(15, 15)
scen = scenarios.scenario_classic
grid.set_reward_states(scen['rewards'])
grid.set_sink_states(scen['sinks'])
mdp = ClassicMDP(Policy(grid), grid)
mdp.load_policy('policies/classic.p')
for i in range(40):
    print grid.step(mdp)

grid.show_recording()
