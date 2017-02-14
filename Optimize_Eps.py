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
		
		self.surr_losses = []

		self.s_l_tf = tf.placeholder('float', shape=[None])
		self.o_e_tf = tf.placeholder('float', shape=[None])

		self.e_tf = self.weight_variable([1])

		self.f_e = tf.multiply(tf.pow(self.e_tf/(self.K-1.0),self.s_l_tf),tf.pow((1.0-self.e_tf),self.T-self.s_l_tf))
		self.f_e_o = tf.sqrt(tf.multiply(tf.pow(self.o_e_tf/(self.K-1.0),self.s_l_tf),tf.pow((1.0-self.o_e_tf),self.T-self.s_l_tf)))

		self.numerator = tf.sqrt(self.f_e)
	

		self.loss = -1.0*tf.reduce_mean(tf.div(self.numerator,self.f_e_o))

		self.train_step = tf.train.MomentumOptimizer(.003, .9)
		self.train = self.train_step.minimize(self.loss)




	def compute_variables(self,trajectories):
		#IPython.embed()
		###
		avg_sur = 0.0
		for traj in trajectories:
			surr_loss = 0.0
			save = True
			for state,label,c_label,r_label,eps in traj:
				if(not c_label == r_label):
					surr_loss += 1.0
				
				

			if(save):
				self.surr_losses.append([surr_loss,eps])
				avg_sur += surr_loss
		print "avg_sur ", avg_sur/float(len(trajectories))
		return zip(*self.surr_losses)


	def grid_search_eps(self,trajectories,disctritzed = 0.1):

		num_eps = int(1.0/disctritzed)

		e_range = np.linspace(0.0,1.0,num=num_eps)

		loss,o_eps = self.compute_variables(trajectories)
		sol = []
		for e in e_range:
			total = 0.0
			for i in range(len(loss)):
				print loss[i]
				num = np.sqrt(((e/(self.K-1.0))**loss[i])*((1.0-e)**(self.T-loss[i])))
				denom = ((o_eps[i]/(self.K-1.0))**loss[i])*((1.0-o_eps[i])**(self.T-loss[i]))

				total += num/denom

			sol.append(total)

		idx = np.argmax(sol)
		print sol

		return e_range[idx]








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







    