import sys
sys.path.insert(0, '../thesis_likelihood')
import numpy as np
# from treestruct import *
# from nodestruct import *
# from prior import *
from buildmtrx import *
from MCMCMoves import *
import random
import csv

import multiprocessing

# random.seed(26111994)

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

N = 5000

debug = 0
t=20

def target1():
	alpha = np.array([0.5,0.5]).astype(object)
	d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda0 = 1, lambda1 = 0.099)
	s = "2A.1T.17C.8G.9G.9T.18A."
	with open('../thesis_likelihood\logs\log1.txt', "w", encoding="utf-8") as f:
		successes, chaina, chainb, chainc = run_chain(s, N, t, Q1, alpha, d, D0, B, Pi, by='df', fname=f, pos=1)

	print("acceptance rate", successes/len(chaina))

	i = 0
	with open(f"../thesis_likelihood\csv\c{i+1}a.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chaina)

	with open(f"../thesis_likelihood\csv\c{i+1}b.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chainb)
		
	with open(f"../thesis_likelihood\csv\c{i+1}c.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chainc)

def target2():
	alpha = np.array([0.5,0.5]).astype(object)
	d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda0 = 1, lambda1 = 0.099)
	s = "(0.15100139284528913 / 'G'(16.33881754412946 / 'A'(3.270634985083973 / 'T'(0.2395460779412808 / 'A')(0.2395460779412808 / 'G'))(3.510181063025252 / 'T'))(19.848998607154712 / 'C'))"
	with open('../thesis_likelihood\logs\log2.txt', "w", encoding="utf-8") as f:
		successes, chaina, chainb, chainc = run_chain(s, N, t, Q1, alpha, d, D0, B, Pi, by='io', fname=f, pos=2)

	print("acceptance rate", successes/len(chaina))

	i = 1
	with open(f"../thesis_likelihood\csv\c{i+1}a.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chaina)

	with open(f"../thesis_likelihood\csv\c{i+1}b.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chainb)
		
	with open(f"../thesis_likelihood\csv\c{i+1}c.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chainc)

def target3():
	alpha = np.array([0.5,0.5]).astype(object)
	d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda0 = 1, lambda1 = 0.099)
	s = "(2.0 / 'A'(8.0 / 'G'(10.0 / 'G')(10.0 / 'T'))(6.747867925981902 / 'T'(11.2521320740181 / 'C')(11.2521320740181 / 'A')))"
	with open('../thesis_likelihood\logs\log3.txt', "w", encoding="utf-8") as f:
		successes, chaina, chainb, chainc = run_chain(s, N, t, Q1, alpha, d, D0, B, Pi, by='io', fname=f, pos=3)

	print("acceptance rate", successes/len(chaina))

	i = 2
	with open(f"../thesis_likelihood\csv\c{i+1}a.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chaina)

	with open(f"../thesis_likelihood\csv\c{i+1}b.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chainb)
		
	with open(f"../thesis_likelihood\csv\c{i+1}c.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chainc)

def target4():
	alpha = np.array([0.5,0.5]).astype(object)
	d, D0, D1, B = build_mtrx(mu0= 0.2, mu1 = 0.099, q01 = 0.2, q10 =0.001, lambda0 = 1, lambda1 = 1)
	s = "(3.616051029646276 / 'A'(8.832241477452936 / 'T'(7.551707492900789 / 'C')(7.471726162738409 / 'G'(0.0799813301623793 / 'G')(0.0799813301623793 / 'T')))(16.383948970353725 / 'A'))"
	with open('../thesis_likelihood\logs\log4.txt', "w", encoding="utf-8") as f:
		successes, chaina, chainb, chainc = run_chain(s, N, t, Q1, alpha, d, D0, B, Pi, by='io', fname=f, pos=4)

	print("acceptance rate", successes/len(chaina))

	i = 3
	with open(f"../thesis_likelihood\csv\c{i+1}a.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chaina)

	with open(f"../thesis_likelihood\csv\c{i+1}b.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chainb)
		
	with open(f"../thesis_likelihood\csv\c{i+1}c.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chainc)

if __name__ == '__main__':
	t1 = multiprocessing.Process(target=target1)
	t2 = multiprocessing.Process(target=target2)
	t3 = multiprocessing.Process(target=target3)
	t4 = multiprocessing.Process(target=target4)

	t1.start()
	t2.start()
	t3.start()
	t4.start()

	t1.join()
	t2.join()
	t3.join()
	t4.join()
