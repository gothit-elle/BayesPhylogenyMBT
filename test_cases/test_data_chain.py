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

i = 101
with open(parentdir + f"/csv/c{i+1}a.csv", newline='') as f:
	reader = csv.reader(f, delimiter='Y')
	data = list(reader)
chaina += data

with open(parentdir + f"/csv/c{i+1}b.csv", newline='') as f:
	data= []
	reader = csv.reader(f,  delimiter='Y')
	for line in reader:
		data.append([float(x) for x in line])
chainb += data

with open(parentdir + f"/csv/c{i+1}c.csv", newline='') as f:
	reader = csv.reader(f,  delimiter='Y')
	data = list(reader)
chainc += data

j = 0
lst = chainb[j]
ind = np.argmax(lst)
s = chaina[j][ind]
t_cur = Tree(1)
t_cur.str2tree(s,20,by='io')
t_cur.disp()
s_ml = t_cur.head.to_newick()

from ete3 import Tree as Tree_n
t = Tree_n(s_ml, format = 3)
# t.convert_to_ultrametric()
# t.show()

print('old tree')
j = 0
lst = chainb[j]
s = chaina[j][0]
t_cur = Tree(1)
t_cur.str2tree(s,20,by='io')
t_cur.disp()
s_start = t_cur.head.to_newick()

t2 = Tree_n(s_start, format = 3)
# t2.convert_to_ultrametric()
# t2.show()

# print(t.write(format = 5))
ret = t.robinson_foulds(t2)
rf, max_rf = ret[0:2]
print(f"RF distance is {rf} over a total of {max_rf}")

print("lik is: ", chainb[j][ind])
print("params are: ", chainc[j][ind])
print('initial lik was: ', chainb[j][0])
print('initial params are: ', chainc[j][0])
N = len(chainb[0])
plt.plot(range(N), chainb[0])
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)
plt.xlabel('Step')
plt.ylabel('Posterior Log-Likelihood')
plt.title("Posterior Log-likelihood vs MCMC Step")
plt.legend(["Chain 1", "Chain 2", "Chain 3", "Chain 4"], loc="lower left")
plt.savefig(parentdir + f"/plots/MCMC_sim{i}.png")