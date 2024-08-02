from sim import * 
from buildmtrx import *
import numpy as np 



import multiprocessing

if __name__ == '__main__':
	stopping_time = 15
	alpha = np.array([0.7,0.3]).astype(object)
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
	
	random.seed(26111994)
	print("beginning simulations...")
	#for i in range(3):
	t1 = sim_tree(alpha, D0, d, B, Q1, Pi, stopping_time, min_leaves = 100, seq_len = 1000)
		# trees[-1].disp()
	print(f"\tgenerated tree with {len(t1.head.find_leaves())} nodes")
	print("simulations done...")
	print("starting MCMC chains...")
	t1 #, t2, t3 = trees
	multiprocessing.freeze_support()
	N = 10000
	t = t1.obs_time
	t1.disp()
	str1 = t1.toStr()
	
	#str2 = t2.toStr()
	#str3 = t3.toStr()

	# print('lik is', log_lik(t2, Q1, Pi, False))
	
	target(str1, N, t, Q1, alpha, d, D0, B, Pi, 101, 1, multip=True)
	#t1 = multiprocessing.Process(target=target, args = (str1, N, t, Q1, alpha, d, D0, B, Pi, 101, 1,))
	#t2 = multiprocessing.Process(target=target, args = (str2, N, t, Q1, alpha, d, D0, B, Pi, 102, 2,))
	#t3 = multiprocessing.Process(target=target, args = (str3, N, t, Q1, alpha, d, D0, B, Pi, 103, 3,))

	#t1.start()
	#t2.start()
	#t3.start()

	#t1.join()
	#t2.join()
	#t3.join()
