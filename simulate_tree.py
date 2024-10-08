from sim import * 
from buildmtrx import *
import numpy as np 
import uuid

import json
import multiprocessing
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 



if __name__ == '__main__':

	stopping_time = 10
	alpha = np.array([0.7,0.3]).astype(object)
	lambda_a = np.array([1, 0.1,0.1,0.1,0.1,0.1,0.1,0.2]).astype(object)
	d, D0, D1, B = build_mtrx(mu0= 0.3, mu1= 0.1, q01 = 0.9, q10 =0.1, lambda_a = lambda_a)
	
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
	t1 = sim_tree(alpha, D0, d, B, Q1, Pi, stopping_time, min_leaves = 50, seq_len = 1000)
	tstamp = str(uuid.uuid4().hex)

	print(t1.head.to_newick(), "\n", tstamp)
	leaves = t1.head.find_leaves()
	mydict = {}
	

	
	for leaf in leaves:
		mydict[leaf.map] = leaf.seq
	with open(currentdir + f"/csv/{tstamp}.json", 'w') as f:
		f.write(json.dumps(mydict)) 
	f.close()
	print(f"\tgenerated tree with {len(leaves)} nodes")
	print("simulations done...")
	print("starting MCMC chains...")
	t1 #, t2, t3 = trees
	multiprocessing.freeze_support()
	N = 20000
	t = t1.obs_time
	print(t)
	# t1.disp()
	str1 = t1 #.toStr()
	
	#str2 = t2.toStr()
	#str3 = t3.toStr()


	
	target(str1, N, t, Q1, alpha, d, D0, B, Pi, 1, multip=True, tstamp = tstamp)
	#t1 = multiprocessing.Process(target=target, args = (str1, N, t, Q1, alpha, d, D0, B, Pi, 101, 1,))
	#t2 = multiprocessing.Process(target=target, args = (str2, N, t, Q1, alpha, d, D0, B, Pi, 102, 2,))
	#t3 = multiprocessing.Process(target=target, args = (str3, N, t, Q1, alpha, d, D0, B, Pi, 103, 3,))

	#t1.start()
	#t2.start()
	#t3.start()

	#t1.join()
	#t2.join()
	#t3.join()
