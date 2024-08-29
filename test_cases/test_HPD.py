import csv
import matplotlib.pyplot as plt
import os
import sys
import inspect
import arviz
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
gparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir) 
sys.path.insert(0, gparentdir) 
import numpy as np
from treestruct import *
from multiprocessing import Pool, freeze_support
import seaborn as sns
from tqdm import tqdm
from itertools import cycle, repeat
import pandas as pd
from ete3 import Tree as Tree_n
from scipy import stats

from lbd_prior import *
from prior import *

TOLER = 5

def read_chains(i):
	data = [] 
	chaina = []
	chainb = []
	chainc = []
	k = 0
	while True:
		try:
			# print("reading file1")
			with open(gparentdir + f"/thesis_likelihood_csv_files/csv/{i}a.csv", newline='') as f:
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
			# print("reading file2")
			with open(gparentdir + f"/thesis_likelihood_csv_files/csv/{i}b.csv", newline='') as f:
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
			# print("reading file3")
			
			with open(gparentdir + f"/thesis_likelihood_csv_files/csv/{i}c.csv", newline='') as f:
				reader = pd.read_csv(f, delimiter='Y', header=None, skiprows=k)
				for n, line in reader.items():
					data.append(line[0])
			chainc += data
			f.close()
			
			break
		except:
			k+=1
	return chaina, chainb, chainc


def find_MBT_stats(chainb, name):
	
	N = len(chainb)
	params = {'mathtext.default': 'regular' }
	plt.plot(range(N), chainb)
	plt.rcParams.update(params)
	plt.ylabel('Posterior Log-Likelihood')
	plt.title("MCMC step")
	plt.savefig(parentdir + f"/plots/MCMC__{name}.png")
	
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
	D0 = np.array([nums[4], nums[5], nums[6], nums[7]])
	B = np.array([(nums[8], nums[9],nums[10],nums[11],nums[12],nums[13],nums[14],nums[15])])
	return [alpha, d, D0, B]

def calc_param_means(chainc):
	alpha_tot = []
	d_tot = []
	D0_tot = []
	B_tot = []
	pool = Pool()
	results1 = pool.map(retrieve_params, chainc)
	pool.close()
	for lst in results1:
		alpha, d, D0, B = lst

		alpha_tot.append(alpha)
		d_tot.append(d)
		D0_tot.append(D0)
		B_tot.append(B)
	
	return alpha_tot, d_tot, D0_tot, B_tot
		
def f(x):
    return x
colours = ["#F72585", "#7209B7", "#3A0CA3", "#4361EE", "#4CC9F0", "#47C9F0"]
palette = cycle(colours)

# p1 = sns.displot(data=df, x='x1', kind='hist', bins=40, stat='density')
def plotfig(x1, name, clear=False, f = f, ptype = sns.histplot, fit=stats.norm):
	ax = ptype(x1,stat='count', binwidth=0.05, color = next(palette) ) #
	params=fit.fit(x1)
	#print(params)
	#xx = np.linspace(*ax.get_xlim(),100)
	#ax.plot(f(xx), f(fit.pdf(xx, *params)), color = next(palette) )
	
	if clear: 
		ax.set_title(f"{name} Frequency")
		ax.figure.savefig(parentdir + f"/plots/{name}.png", bbox_inches='tight')
		ax.figure.clf()
	
def plot_arr(array, hash):
	array = np.transpose(np.array(array))
	N = len(array)
	for i in range(N): 
		plotfig(array[i], f"{hash}_{i}", clear = (i==N-1))
		

		
def MLE_lbd(chaina, chainb, chainc, alpha, d, D0, B, rf_lbd):
	max = 0
	idx_max = 0
	# take a set of unique things to calculate, take 4 trees her
	ind = np.argpartition(chainb, -100)[-100:]
	N = len(ind)
	for j in tqdm(range(N)):
		i = ind[j]
		if i>0 and chainb[i] == chainb[i-1]:
			pass
		else:
			tree = Tree(1)
			tree.str2tree(chaina[i],20,by='io')
			tree.obs_time = tree.head.find_max_dist()
			# calc the LBD prior
			lambda0 = np.average(B[i])
			mu0 = np.average(d[i])

			lbd_p = lin_bd_lik(lambda0, mu0, tree) #bd_lik(tree, lambda0, mu0, log=True, adj = True)
			# calc the MBT prior
			mbt_p = tree_prior(tree, np.array(alpha[i]).astype(object), np.array(d[i]).astype(object), np.array(D0[i]).astype(object).reshape(2,2), np.array(B[i]).astype(object).reshape(2,4), logit=True, fname = None, multip=True)
			# regenerate the likelihood by doing posterior - MBT prior
			# get the LBD posterior

			lbd_post = chainb[i] - mbt_p + lbd_p
			
			if max == 0: 
				max = lbd_post
				idx_max = i
			if max < lbd_post: 
				max = lbd_post
				idx_max = i
	
	print("MAX via LBD is: ", max)
	ml_tree = Tree(1)
	ml_tree.str2tree(chaina[1000:][idx_max],20,by='io')
	ml_tree.obs_time = ml_tree.head.find_max_dist()
	s_start = tree.head.to_newick()
	t = Tree_n(s_start, format = 3)
	s_ml = ml_tree.head.to_newick()
	t2 = Tree_n(s_ml, format = 3)
	# t.show()
	# t2.show()


	ret = t.robinson_foulds(t2)
	rf, max_rf = ret[0:2]
	print(f"\tRF distance is {rf} over a total of {max_rf}")
	rf_lbd.append(rf/max_rf)



	return rf_lbd
def plot_stats(init_tree, ml_tree, chainc, chaina, ind, rf_totals, alpha, d, D0, B):
	# print("\t True ", init_tree.obs_time, ml_tree.obs_time )
	s_start = init_tree.head.to_newick()
	t = Tree_n(s_start, format = 3)
	s_ml = ml_tree.head.to_newick()
	t2 = Tree_n(s_ml, format = 3)
	# t.show()
	# t2.show()
	mlpc = find_pop_curve(s)
	tpc = find_pop_curve(chaina[0])


	ret = t.robinson_foulds(t2)
	rf, max_rf = ret[0:2]
	print(f"\tRF distance is {rf} over a total of {max_rf}")
	rf_totals.append(rf/max_rf)
	leaves = init_tree.head.find_leaves()
	seqlen = len(leaves[0].seq)
	
	print(f"\tTree has {len(leaves)} leaves and a seqlen of {seqlen}")

	"""
	pool = Pool()
	results = pool.starmap(plot_arr, zip([alpha, d, D0, B], repeat(i))) 
	pool.close()
	

	pool = Pool()
	results = pool.map(find_pop_curve, chaina[1000:]) # remove burn in
	pool.close()
	times = np.linspace(0, init_tree.obs_time, 50)
	mean_pops = sum(results)/len(results)


		
	plt.plot(times, mean_pops, label="mean pop curve")
	plt.plot(times, mlpc, label="most likely pop curve")
	plt.plot(times, tpc, label="true pop curve")
	params = {'mathtext.default': 'regular' }
	plt.rcParams.update(params)
	plt.xlabel('t')
	plt.ylabel('population size')
	plt.title("population size over time")
	plt.legend()
	plt.savefig(f"../thesis_likelihood/plots/meanpop_{i}.png")
	print("\n")
	plt.clf()
	"""
	return rf_totals


if __name__ == '__main__':
	freeze_support()
	# list to store files
	hashes = []
	rf_totals = []
	rf_lbd = []
	# Iterate directory
	for path in os.listdir(gparentdir + f"/thesis_likelihood_csv_files/csv/"):
		# check if current path is a file
		if os.path.isfile(os.path.join(gparentdir + "/thesis_likelihood_csv_files/csv/", path)):
			if path[-5] == 'a':
				hashes.append(path[0:-5])
	print(hashes)
	#i = 1 #"3929977c72a745b180364a64a6c3a74c"
	# i = "c056e72282e340e797642326d209c601" #this reqs skip rows
	
	for j in range(len(hashes)):
		i = hashes[j]
		chaina, chainb, chainc = read_chains(i)
		
		
		
		
		if len(chainb) >= 3000:
			print("\nStep ", j, ", for dataset: ", i, "len", len(chainb))
			lst = chainb[1000:] 
			ind = np.argmax(lst)
			s = chaina[1000:][ind]
			
			
			init_tree = Tree(1)
			init_tree.str2tree(chaina[0],20,by='io')
			init_tree.obs_time = init_tree.head.find_max_dist()
			

			# t2.convert_to_ultrametric()

			
			ml_tree = Tree(1)
			ml_tree.str2tree(chaina[1000:][ind],20,by='io')
			ml_tree.obs_time = ml_tree.head.find_max_dist()

			# t2.convert_to_ultrametric()

			
			if init_tree.obs_time + TOLER <  ml_tree.obs_time or init_tree.obs_time - TOLER > ml_tree.obs_time:
				print("\t FALSE ", init_tree.obs_time, ml_tree.obs_time )
			else:
				print("\t TRUE ", init_tree.obs_time, ml_tree.obs_time )
				alpha, d, D0, B = calc_param_means(chainc[1000:]) # remove burn in
				N = len(chainc[1000:]) 
				# find_MBT_stats(chainb, i)
				print("\talpha: ", sum(alpha)/N)
				print("\td: ", sum(d)/N)
				print("\tD0: ", sum(D0)/N)
				print("\tB: ", sum(B)/N)


						
						
				print("\t ml alpha: ", alpha[ind])
				print("\t ml d: ", d[ind])
				print("\t ml D0: ", D0[ind])
				print("\t ml B: ", B[ind])
				rf_totals = plot_stats(init_tree, ml_tree, chainc, chaina, ind, rf_totals, alpha, d, D0, B)
				# rf_lbd = MLE_lbd(chaina[1000:], chainb[1000:], chainc[1000:], alpha, d, D0, B, rf_lbd)
				
				
	plotfig(np.array(rf_totals), "normalised rf distance", clear=True)
	#plotfig(np.array(rf_lbd), "normalised rf distance with LBD prior", clear=True)
	#print("sample size = ", len(rf_totals))

	#print(find_pop_curve(t_cur))




