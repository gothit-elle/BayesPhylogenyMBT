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

chaina = []
chainb = []
chainc = []


import pandas as pd
data = [] 
i = "6201d47f12de48fbab3156d1c62562fc"
# i = "c056e72282e340e797642326d209c601"
with open(parentdir + f"/csv/c{i}a.csv", newline='') as f:
	reader = pd.read_csv(f, delimiter='Y', header=None, skiprows=0)
	for n, line in reader.iteritems():
		data.append(line[0])
chaina += data
data = []
with open(parentdir + f"/csv/c{i}b.csv", newline='') as f:
	data= []
	reader = pd.read_csv(f, delimiter='Y', header=None, skiprows=0)
	for n, line in reader.iteritems():
		data.append(float(line[0]))
chainb += data
data = []
print(chainb)
with open(parentdir + f"/csv/c{i}c.csv", newline='') as f:
	reader = pd.read_csv(f, delimiter='Y', header=None, skiprows=0)
	for n, line in reader.iteritems():
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