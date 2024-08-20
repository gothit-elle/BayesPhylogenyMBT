import csv
import matplotlib.pyplot as plt
import os
import sys
import inspect
import arviz
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import numpy as np
from treestruct import *
from multiprocessing import Pool, freeze_support
from tqdm import tqdm

import pandas as pd
from ete3 import Tree as Tree_n


def read_chains(i):
	data = [] 
	chaina = []
	chainb = []
	chainc = []
	k = 0
	while True:
		try:
			with open(parentdir + f"/csv/c{i}a.csv", newline='') as f:
				reader = pd.read_csv(f, delimiter='Y', header=None, skiprows=k)
				for n, line in reader.items():
					data.append(line[0])
			chaina += data
			f.close()
			break
		except:
			k += 1
	data = []
	k=0
	while True:
		try:
			with open(parentdir + f"/csv/c{i}b.csv", newline='') as f:
				data= []
				reader = pd.read_csv(f, delimiter='Y', header=None, skiprows=k)
				for n, line in reader.items():
					data.append(float(line[0]))
			chainb += data
			f.close()
			break
		except:
			k+=1
	data = []
	k=0
	while True:
		try:
			with open(parentdir + f"/csv/c{i}c.csv", newline='') as f:
				reader = pd.read_csv(f, delimiter='Y', header=None, skiprows=k)
				for n, line in reader.items():
					data.append(line[0])
			chainc += data
			f.close()
			
			break
		except:
			k+=1
	return chaina, chainb, chainc


def find_MBT_stats(i):
	chaina, chainb, chainc = read_chains(i)
	N = len(chainb)
	print(arviz.hdi(np.array(chainb)))
	plt.plot(range(N), chainb)
	params = {'mathtext.default': 'regular' }
	plt.rcParams.update(params)
	plt.ylabel('Posterior Log-Likelihood')
	plt.title("Sorted MCMC results")
	plt.savefig(parentdir + f"/plots/MCMC_sorted_{i}.png")
	
def find_pop_curve(str):
	tree = Tree(1)
	tree.str2tree(str,20,by='io')
	tree.obs_time = tree.head.find_max_dist()
	times = np.linspace(0, tree.obs_time, 50)

	pops = [find_pop(tree.head, time) for time in times]
	return np.array(pops)
	
def find_pop(cur, stop_time):
	n = 0

	if cur.time < stop_time:
		if cur.isLeaf(): 
			return 1
		n += find_pop(cur.right, stop_time - cur.time) + find_pop(cur.left, stop_time - cur.time)
	else:
		n = 1
	return n
#find_MBT_stats(chaina, chainb, chainc)

def retrieve_params(str):
	nums = re.findall('\d+.\d+', str)
	nums = [float(num) for num in nums]
	alpha = np.array([nums[0], nums[1]])
	d = np.array([nums[2], nums[3]])
	D0 = np.array([nums[4], nums[5], nums[6], nums[7]]).reshape(2,2).astype(object)
	B = np.array([(nums[8], nums[9],nums[10],nums[11],nums[12],nums[13],nums[14],nums[15])]).reshape(2,4).astype(object)
	return [alpha, d, D0, B]

def calc_param_means(chainc):
	alpha_tot = np.array([0, 0]).astype(object)
	d_tot = np.array([0, 0]).astype(object)
	D0_tot = np.array([0, 0, 0, 0]).reshape(2,2).astype(object)
	B_tot = np.array([(0, 0,0,0,0,0,0,0)]).reshape(2,4).astype(object)
	pool = Pool()
	results1 = pool.map(retrieve_params, chainc)
	pool.close()
	for lst in results1:
		alpha, d, D0, B = lst

		alpha_tot += alpha
		d_tot += d
		D0_tot += D0
		B_tot += B
	N = len(chainc)
	return alpha_tot/N, d_tot/N, D0_tot/N, B_tot/N
		
		
if __name__ == '__main__':
	freeze_support()
	# list to store files
	hashes = []

	# Iterate directory
	for path in os.listdir(parentdir + "/csv/"):
		# check if current path is a file
		if os.path.isfile(os.path.join(parentdir + "/csv/", path)):
			if path[-5] == 'a':
				hashes.append(path[1:-5])
	print(hashes)
	#i = 1 #"3929977c72a745b180364a64a6c3a74c"
	# i = "c056e72282e340e797642326d209c601" #this reqs skip rows
	
	for j in range(len(hashes)):
		i = hashes[j]
		chaina, chainb, chainc = read_chains(i)

		
		print("\nStep ", j, ", for dataset: ", i, "len", len(chainb))
		if len(chainb) >= 10000 and i != "c056e72282e340e797642326d209c601":
			lst = chainb[1000:]
			ind = np.argmax(lst)
			s = chaina[1000:][ind]

			mlpc = find_pop_curve(s)
			tpc = find_pop_curve(chaina[0])
			alpha, d, D0, B = calc_param_means(chainc[1000:]) # remove burn in
			print("\talpha: ", alpha)
			print("\td: ", d)
			print("\tD0: ", D0)
			print("\tB: ", B)
			
			alpha, d, D0, B = retrieve_params(chainc[1000:][ind]) # remove burn in
			print("\t ml alpha: ", alpha)
			print("\t ml d: ", d)
			print("\t ml D0: ", D0)
			print("\t ml B: ", B)
			
			pool = Pool()
			results = pool.map(find_pop_curve, chaina[1000:]) # remove burn in
			pool.close()
			
			tree = Tree(1)
			tree.str2tree(chaina[0],20,by='io')
			tree.obs_time = tree.head.find_max_dist()
			"""
			s_start = tree.head.to_newick()
			t = Tree_n(s_start, format = 3)
			# t2.convert_to_ultrametric()
			t.show()
			"""
			leaves = tree.head.find_leaves()
			seqlen = len(leaves[0].seq)
			
			print(f"\tTree has {len(leaves)} leaves and a seqlen of {seqlen}")
			times = np.linspace(0, tree.obs_time, 50)
			mean_pops = sum(results)/len(results)
			"""
			tree = Tree(1)
			tree.str2tree(chaina[1000:][ind],20,by='io')
			tree.obs_time = tree.head.find_max_dist()
			s_start = tree.head.to_newick()
			t2 = Tree_n(s_start, format = 3)
			# t2.convert_to_ultrametric()
			t2.show()
			
			ret = t.robinson_foulds(t2)
			rf, max_rf = ret[0:2]
			print(f"\tRF distance is {rf} over a total of {max_rf}")
			"""		
			plt.plot(times, mean_pops, label="mean pop curve")
			plt.plot(times, mlpc, label="most likely pop curve")
			plt.plot(times, tpc, label="true pop curve")
			params = {'mathtext.default': 'regular' }
			plt.rcParams.update(params)
			plt.xlabel('t')
			plt.ylabel('population size')
			plt.title("population size over time")
			plt.legend()
			plt.savefig(f"../thesis_likelihood/plots/meanpop_{j}.png")
			print("\n")
			plt.clf()
		

	#print(find_pop_curve(t_cur))





