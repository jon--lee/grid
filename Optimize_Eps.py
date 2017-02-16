''''
This code computes the convex problem to solve for sample estimat of the best 
noise term. 
'''

import tensorflow as tf
from numpy.random import randint
import IPython
import math
import numpy as np


class INIT_OPT(): 

	def __init__(self,T,K):
		self.T = float(T)
		self.K = float(K)
		
		


	def compute_avg_case(self,trajectories):
		surr_losses = []
		avg_sur = 0.0
		w_loss = 0.0
		for traj in trajectories:
			T = len(traj)
			surr_loss = 0.0
			save = True
			for state,label,c_label,r_label,eps in traj:
				if(not c_label == r_label):
					surr_loss += 1.0
				
				

			if(save):
				surr_losses.append([surr_loss,eps])
				avg_sur += surr_loss
				if(w_loss < surr_loss):
					w_loss = surr_loss
				print "SURR LOSS ",surr_loss
		print "TIME HORIZON ",T
		print "AVG_SUR ", avg_sur/float(len(trajectories))
		#print "WORST CASE LOSS ",w_loss
		return avg_sur/float(len(trajectories))#w_loss


	def compute_variables(self,trajectories):
		#IPython.embed()
		###
		surr_losses = []
		avg_sur = 0.0
		for traj in trajectories:
			T = len(traj)
			surr_loss = 0.0
			save = True
			for state,label,c_label,r_label,eps in traj:
				if(not c_label == r_label):
					surr_loss += 1.0
				
				

			if(save):
				surr_losses.append([surr_loss,eps])
				avg_sur += surr_loss
				#print "SURR LOSS ",surr_loss
		print "TIME HORIZON ",T
		print "AVG_SUR ", avg_sur/float(len(trajectories))
		return zip(*surr_losses)


	def grid_search_eps(self,trajectories,disctritzed = 0.1):

		num_eps = int(1.0/disctritzed)

		e_range = np.linspace(0.0,1.0,num=num_eps)

		loss,o_eps = self.compute_variables(trajectories)

		avg_loss = self.compute_avg_case(trajectories)#+self.T*1.0/np.sqrt(20)

		sol = []
		for e in e_range:

			total = np.sqrt(((e/(self.K-1.0))**avg_loss)*((1.0-e)**(self.T-avg_loss)))
			sol.append(total)

		best_eps = np.max(sol)

		#upper bound on robot's distributino 


		# for e in e_range:
		# 	total = 0.0
		# 	for i in range(len(loss)):
				
		# 		num = ((e/(self.K-1.0))**loss[i])*((1.0-e)**(self.T-loss[i]))

		# 		#denom = ((o_eps[i]/(self.K-1.0))**loss[i])*((1.0-o_eps[i])**(self.T-loss[i]))

		# 		total += num

		# 	sol.append(total)
			

		idx = np.argmax(sol)
		print sol
		print "PLAYING ",e_range[idx]

		return 0.1#e_range[idx]








if __name__ == '__main__':

	trajs = 100

	control_space = [0,1]

	T = 70
	K = 5

	trajectories = []
	for m in range(trajs):
		traj = []
		for t in range(T):
			label = randint(K)
			sup_lbl = randint(K)
			robot_lbl = randint(K)

			state = 0.0

			traj.append([state,label,sup_lbl,robot_lbl,0.5])

		trajectories.append(traj)


	init = INIT_OPT(T,K)

	eps = init.grid_search_eps(trajectories)

	print 'Best EPS ',eps







    