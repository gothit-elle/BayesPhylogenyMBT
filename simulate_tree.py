from sim import * 
from buildmtrx import *
import numpy as np 

def main():
	time = 7
	alpha = np.array([0.5,0.5]).astype(object)
	d, D0, D1, B = build_mtrx(mu0= 0.15, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda0 = 1, lambda1 = 0.099)

	
	# we start a tree
	# relative rates matrix
	R = np.array([0, 0.5, 2, 0.5,
				  0.5, 0, 0.5, 2,
				  2, 0.5, 0, 0.5,
				  0.5, 2, 0.5, 0]).reshape(4,4)
	# initial distribution
	Pi = [0.295, 0.205, 0.205, 0.295]
	# rate matrix
	Q1 = R@np.diag(Pi)
	# need to adjust so rows sum to 0
	Q1 -= np.diag(Q1@np.ones(4))
	
	trees = []
	for i in range(3):
		trees.append(sim_tree_i(alpha, D0, d, B, Q1, Pi, time))
	
	return trees[0], trees[1], trees[2], Q1, Pi, alpha, d, D0, D1, B

def sim_tree_i(alpha, D0, d, B, Q1, Pi, time):
	t_next = sim_tree(alpha, D0, d, B, Q1, Pi, time)
	t_next.obs_time = time

	while t_next.head.right == None or t_next.head.left == None:
		t_next = sim_tree(alpha, D0, d, B, Q1, Pi, time)
		t_next.obs_time = time
	t_next.disp()
	return t_next	

import multiprocessing

if __name__ == '__main__':
	random.seed(26111994)
	t1, t2, t3, Q1, Pi, alpha, d, D0, D1, B = main()
	multiprocessing.freeze_support()
	N = 5000
	t = t2.obs_time
	str1 = t1.toStr()
	str2 = t2.toStr()
	str3 = t3.toStr()
	#print('lik is', log_lik(t2, Q1, Pi, False))
	
	t1 = multiprocessing.Process(target=target, args = (str1, N, t, Q1, alpha, d, D0, B, Pi, 101, 1,))
	t2 = multiprocessing.Process(target=target, args = (str2, N, t, Q1, alpha, d, D0, B, Pi, 102, 2,))
	t3 = multiprocessing.Process(target=target, args = (str3, N, t, Q1, alpha, d, D0, B, Pi, 103, 3,))

	t1.start()
	t2.start()
	t3.start()

	t1.join()
	t2.join()
	t3.join()
