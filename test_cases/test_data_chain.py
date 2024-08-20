import csv
import matplotlib.pyplot as plt
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import numpy as np
from treestruct import *
from posterior import *
from buildmtrx import *

chaina = []
chainb = []
chainc = []


import pandas as pd
data = [] 
i = "c056e72282e340e797642326d209c601"
# i = "c056e72282e340e797642326d209c601" #this reqs skip rows
with open(parentdir + f"/csv/c{i}a.csv", newline='') as f:
	reader = pd.read_csv(f, delimiter='Y', header=None, skiprows=9)
	for n, line in reader.items():
		data.append(line[0])
chaina += data
data = []
with open(parentdir + f"/csv/c{i}b.csv", newline='') as f:
	data= []
	reader = pd.read_csv(f, delimiter='Y', header=None, skiprows=8)
	for n, line in reader.items():
		data.append(float(line[0]))
chainb += data
data = []

with open(parentdir + f"/csv/c{i}c.csv", newline='') as f:
	reader = pd.read_csv(f, delimiter='Y', header=None, skiprows=8)
	for n, line in reader.items():
		data.append(line[0])
chainc += data

j = 0
lst = chainb
ind = np.argmax(lst)
print(len(chainb))
s = chaina[ind]
print(s)
t_cur = Tree(1)
t_cur.str2tree(s,20,by='io')
t_cur.disp()
s_ml = t_cur.head.to_newick()

from ete3 import Tree as Tree_n
t = Tree_n(s_ml, format = 3)
# t.convert_to_ultrametric()
t.show()

print('old tree')
j = 0
lst = chainb
s = chaina[0]
t_cur = Tree(1)
t_cur.str2tree(s,20,by='io')
t_cur.disp()
s_start = t_cur.head.to_newick()
t2 = Tree_n(s_start, format = 3)
# t2.convert_to_ultrametric()
t2.show()

# print(t.write(format = 5))
ret = t.robinson_foulds(t2)
rf, max_rf = ret[0:2]
print(f"RF distance is {rf} over a total of {max_rf}")

print("lik is: ", chainb[ind])
print("params are: ", chainc[ind])


alpha = np.array([0.9705533474125201, 0.02944665258747998]).astype(object)
d, D0, D1, B = build_mtrx(mu0= 0.2977797601236012, mu1=0.1493045447617091, q01 =  0.7728589418470766, q10 =0.017171938988586648, lambda0 = 1, lambda1 = 0.099)
s = chaina[ind]
t_cur = Tree(1)
t_cur.str2tree(s,20,by='io')

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

	
print(tree_posterior(t_cur, alpha, d, D0, B, Q1, Pi, debug=False, fname = None, multip=False))
print('initial lik was: ', chainb[0])
print('initial params are: ', chainc[0])
N = len(chainb)
plt.plot(range(N), chainb)
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)
plt.xlabel('Step')
plt.ylabel('Posterior Log-Likelihood')
plt.title("Posterior Log-likelihood vs MCMC Step")
plt.legend(["Chain 1", "Chain 2", "Chain 3", "Chain 4"], loc="lower left")
plt.savefig(parentdir + f"/plots/MCMC_sim{i}.png")