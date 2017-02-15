# full trial (overnight)
#ITER = 45
#TRIALS = 10
#SAMP = 15
    
# partial trial (several hours when number of scenarios is limited)
ITER = 15
TRIALS = 10
SAMP = 10

INIT_SAMPLES = 20




# debugging     (several minutes)
#ITER = 2
#TRIALS = 1
#SAMP = 2

# specify settings (i.e. you can add moves = [30, 70] #to get tests with 30 and 70 moves)
ld_set = [5] #(limit_data) number of trajectories per iteration added to dataset
d_set = [-1] #depth of decision tree (-1 for LinearSVC)
moves = [70] #number of steps in each trajectory
# p_beta_set = [0.0, .3]            #starting beta parameter for dagger
# p_beta_set = [.5, .8, 1.0]
# p_beta_set = [.1, .2]
# p_beta_set = [.4, .6]
# p_beta_set = [.7, .9]
p_beta_set = [0.0]

