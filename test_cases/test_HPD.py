import csv
import matplotlib.pyplot as plt
import os
import sys
import inspect
from kendall_colijn import *

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
import phylotreelib as pt
from phylotreelib import *

from prior import *
from scipy.linalg import expm

TOLER = 5
BURNIN = 10000
look_in = gparentdir + "/thesis_likelihood_csv_files/csv/"
LIMIT = 30
STEPS = 100
#look_in = parentdir + "/csv/"

# this code is taken from phylotreelib and updated to work with our input
def my_set_ca_node_depths(self, sum_tree, wt_count_burnin_filename_list):
	"""Set branch lengths on summary tree based on mean node depth for clades corresponding
	to MRCA of clade's leaves. (same as "--height ca" in BEAST's treeannotator)
	This means that all input trees are used when computing
	mean for each node (not just the input trees where that exact monophyletic clade
	is present)"""

	# Initialize node dictionary: {nodeid:Nodestruct}. Keeps track of avg depths
	nodedict = {}
	for node in sum_tree.nodes:
		nodedict[node] = Nodestruct(depth = 0.0)

	# Find mean common ancestor depth for all internal nodes
	# (I assume input trees are from clock models, so leaf-depths are constant)
	wsum = 0.0
	for weight, count, burnin, chaina in wt_count_burnin_filename_list:
		ntrees = count - burnin
		wsum += weight
		multiplier = weight / ntrees
		
		for str in chaina:
			str = format_str(str)
			input_tree = pt.Tree.from_string(str)
			for node in sum_tree.intnodes:
				sumt_remkids = sum_tree.remotechildren_dict[node]
				input_mrca = input_tree.find_mrca(sumt_remkids)
				input_depth = input_tree.nodedepthdict[input_mrca]
				nodedict[node].depth += input_depth * multiplier

	# normalise values for internal nodes by sum of weights
	for node in sum_tree.intnodes:
		nodedict[node].depth /= wsum

	# Set values for leaves
	# Use values on last tree left over from looping above (assume same on all input trees)
	for node in sum_tree.leaves:
		nodedict[node].depth = input_tree.nodedepthdict[node]

	# use average depths to set branch lengths
	for parent in sum_tree.sorted_intnodes(deepfirst=True):
		p_depth = nodedict[parent].depth
		for child in sum_tree.children(parent):
			c_depth = nodedict[child].depth
			blen = p_depth - c_depth
			sum_tree.setlength(parent, child, blen)

	return sum_tree

def read_chains(i):
	data = [] 
	chaina = []
	chainb = []
	chainc = []
	chaind = []
	k = 0
	while True:
		try:
			# print("reading file1")
			with open(look_in + f"{i}a.csv", newline='') as f:
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
			with open(look_in + f"{i}b.csv", newline='') as f:
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
			
			with open(look_in+ f"{i}c.csv", newline='') as f:
				reader = pd.read_csv(f, delimiter='Y', header=None, skiprows=k)
				for n, line in reader.items():
					data.append(line[0])
			chainc += data
			f.close()
			
			break
		except:
			k+=1
	data = []
	k = 0

	while True:
		try:
			# print("reading file2")
			with open(look_in + f"{i}d.csv", newline='') as f:
				data= []
				reader = pd.read_csv(f, delimiter='Y', header=None, skiprows=k)
				for n, line in reader.items():
					data.append(float(line[0]))
			chaind += data
			f.close()
			break
		except:
			k+=1
	return chaina, chainb, chainc, chaind


def find_MBT_stats(chainb, name):
	
	N = len(chainb)
	params = {'mathtext.default': 'regular' }
	plt.plot(range(N), chainb)
	plt.rcParams.update(params)
	plt.ylabel('Posterior Log-Likelihood')
	plt.title("MCMC step")
	plt.savefig(parentdir + f"/plots/MCMC__{name}.png")
	
def find_pop_curve(str, by='nw'):
	if by != "TR":
		tree = Tree(1)
		tree.str2tree(str,20,by=by)
		tree.obs_time = tree.head.find_max_dist()
	else:
		tree = str # tree passed in
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
	D0 = np.array([-nums[4], nums[5], nums[6], -nums[7]])
	B = np.array([nums[8], nums[9],nums[10],nums[11],nums[12],nums[13],nums[14],nums[15]])
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
#palette = cycle(colours)

# p1 = sns.displot(data=df, x='x1', kind='hist', bins=40, stat='density')
def plotfig(x1, name, pname, clear=False, f = f, ptype = sns.boxplot, fit=stats.norm):
	ax = ptype(x1, palette='pastel') #,stat='count',  binwidth=0.05,
	params=fit.fit(x1)
	#print(params)
	#xx = np.linspace(*ax.get_xlim(),100)
	#ax.plot(f(xx), f(fit.pdf(xx, *params)), color = next(palette) )
	
	if clear: 
		ax.set_title(f"{pname} boxplot")
		ax.figure.savefig(parentdir + f"/plots/{name}.png", bbox_inches='tight')
		ax.figure.clf()
	
def plot_arr(array, hash, name):
	#array = np.transpose(np.array(array))
	N = len(array)
	plotfig(pd.DataFrame(array), f"{hash}_{name}",name, clear = True)
	#for i in range(N): 
	#	plotfig(array[i], f"{hash}_{i}", clear = (i==N-1))
		

def TTE(alpha, d, D0, B):
	alpha = alpha.astype(object)
	D0 = D0.astype(object)
	B = B.astype(object)
	d = d.astype(object)
	res = get_E(LIMIT, alpha, d, D0.reshape(2,2), B.reshape(2,4), plot=1).y
	print(res)
	return alpha@res


def MTPS(time, alpha, D0, B):
	alpha = alpha.astype(object)
	D0 = D0.astype(object)
	B = B.astype(object)
	
	times = np.linspace(0,time)
	N = len(B.reshape(2,4))
	prod = np.transpose(np.kron(np.ones(N),np.identity(N)) + np.kron(np.identity(N),np.ones(N))) # this transpose is once again sus

	omega = D0.reshape(2,2) + B.reshape(2,4)@(prod)

	return [alpha@expm(omega*t)@np.ones(N) for t in times]
		
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
	ml_tree.str2tree(chaina[BURNIN:][idx_max],20,by='io')
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
	

def plot_stats(init_tree, mlt_str, lbdt_str, mcc_tree, chaina, ind_mle, ind_lbd, rf_totals_map, rf_totals_lbd, rf_totals_mcc, alpha, d, D0, B):
	# print("\t True ", init_tree.obs_time, ml_tree.obs_time )
	s_start = init_tree.head.to_newick()
	t = Tree_n(s_start, format = 3)

	t2 = Tree_n(mlt_str, format = 3)
	
	t3 = Tree_n(lbdt_str, format=3)
	
	s_mcc = mcc_tree.head.to_newick()
	t4 = Tree_n(s_mcc, format = 3)
	# t.show()
	# t2.show()
	mlpc = find_pop_curve(mlt_str, by='nw')
	tpc = find_pop_curve(chaina[0], by='io')
	lbdpc = find_pop_curve(lbdt_str, by='nw')
	# mccpc = find_pop_curve(mcc_tree, by='TR') # this tree has no branch lengths
	# mcc_tree.disp()
	def rf_dist(t, t2, name): 
		ret = t.robinson_foulds(t2)
		rf, max_rf = ret[0:2]
		print(f"\t{name} RF distance is {rf} over a total of {max_rf}")
		return rf/max_rf
	rf_totals_map.append(rf_dist(t, t2, "MAP Tree"))
	rf_totals_lbd.append(rf_dist(t, t3, "LBD Tree"))
	rf_totals_mcc.append(rf_dist(t, t4, "MCC Tree"))
	
	
	
	leaves = init_tree.head.find_leaves()
	seqlen = len(leaves[0].seq)
	
	print(f"\tTree has {len(leaves)} leaves and a seqlen of {seqlen}")

	
	pool = Pool()
	results = pool.starmap(plot_arr, zip([alpha, d, D0, B], repeat(i), ["alpha", "d", "D0", "B"])) 
	pool.close()
	
	
	pool = Pool()
	results = pool.map(find_pop_curve, chaina[BURNIN:]) # remove burn in
	pool.close()
	times = np.linspace(0, init_tree.obs_time, 50)
	mean_pops = sum(results)/len(results)

	
	mean_total_pop_true = MTPS(init_tree.obs_time, alpha[0], D0[0], B[0])
	extinction_time_true = TTE(alpha[0], d[0], D0[0], B[0])
	
	mean_total_pop_MAP = MTPS(init_tree.obs_time, alpha[BURNIN:][ind_mle], D0[BURNIN:][ind_mle], B[BURNIN:][ind_mle])
	extinction_time_MAP = TTE(alpha[BURNIN:][ind_mle], d[BURNIN:][ind_mle], D0[BURNIN:][ind_mle], B[BURNIN:][ind_mle])
		
	#mean_total_pop_LBD = MTPS(alpha[BURNIN:][ind_LBD], D0[BURNIN:][ind_LBD], B[BURNIN:][ind_LBD].reshape(2,4))
	#extinction_time_LBD = TTE(alpha[BURNIN:][ind_LBD], d[BURNIN:][ind_LBD], D0[BURNIN:][ind_LBD], B[BURNIN:][ind_LBD].reshape(2,4))
	

	# plt.plot(times, mccpc, label="MCC pop curve")
	params = {'mathtext.default': 'regular' }
	plt.rcParams.update(params)
	plt.xlabel('t')
	plt.ylabel('Population size')
	plt.title("Empirical Population Size Over Time")
	plt.plot(times, mean_pops, label="Mean Population Curve")
	plt.plot(times, mlpc, label="MBT MAP tree Population Curve")
	
	plt.plot(times, tpc, label="true Population Curve")
	plt.plot(times, lbdpc, label="LBD MAP Population Curve")
	plt.legend()
	plt.savefig(f"../thesis_likelihood/plots/meanpop_{i}.png")
	plt.clf()
	

	params = {'mathtext.default': 'regular' }
	plt.rcParams.update(params)
	plt.xlabel('t')
	plt.ylabel('M(t)')
	plt.title("Theoretical Mean Total Population Size")
	plt.plot(np.linspace(0,LIMIT), mean_total_pop_true, label="True")
	plt.plot(np.linspace(0,LIMIT), mean_total_pop_MAP, label="MAP")
	plt.legend()
	plt.savefig(f"../thesis_likelihood/plots/meanpop_theoretical_{i}.png")
	plt.clf()
	

	params = {'mathtext.default': 'regular' }
	plt.rcParams.update(params)
	plt.xlabel('t')
	plt.ylabel(r'$\alpha \mathbf{E}(t)$')
	plt.title("Theoretical Time to Extinction")
	plt.plot(np.linspace(0,LIMIT,STEPS), extinction_time_true, label="True")
	plt.plot(np.linspace(0,LIMIT,STEPS), extinction_time_MAP, label="MAP")
	plt.legend()
	plt.savefig(f"../thesis_likelihood/plots/extinction_theoretical_{i}.png")
	plt.clf()
	
	return rf_totals_map, rf_totals_lbd, rf_totals_mcc

def format_str(str):
	str = str.replace("None", '')
	N = len(str)
	for j in range(N):
		i = str[N - j - 1]
		if i == ':':
			str = str[:N-j-1]
			str += ";"
			return str
	

if __name__ == '__main__':
	freeze_support()
	# list to store files
	hashes = []
	rf_totals = []
	rf_lbd = []
	# Iterate directory

	for path in os.listdir(look_in):
		# check if current path is a file
		if os.path.isfile(os.path.join(look_in, path)):
			if path[-5] == 'a':
				hashes.append(path[0:-5])
	print(hashes)
	#i = 1 #"3929977c72a745b180364a64a6c3a74c"
	# i = "c056e72282e340e797642326d209c601" #this reqs skip rows
	
	for j in range(len(hashes)):
		i = hashes[j]
		chaina, chainb, chainc, chaind = read_chains(i)
		
		
		
		
		if len(chainb) >= 10002:
			print("\nStep ", j, ", for dataset: ", i, "len", len(chainb))
			
			# check convergence rq
			N = len(chainb)
			plt.plot(range(N), chainb)
			params = {'mathtext.default': 'regular' }
			plt.rcParams.update(params)
			plt.xlabel('Step')
			plt.ylabel('Posterior Log-Likelihood')
			plt.title("Posterior Log-likelihood vs MCMC Step")

			plt.savefig(parentdir + f"/plots/MCMC_sim_{i}.png")
			plt.clf()
			
			lst = chainb[BURNIN:] 
			ind = np.argmax(lst)
			s = chaina[BURNIN:][ind]
			treesummary = pt.BigTreeSummary(trackbips=False, trackclades=True, trackroot=True)
			for str in tqdm(chaina[BURNIN:]):
				str = format_str(str)
				mytree = pt.Tree.from_string(str)
				treesummary.add_tree(mytree)
			mcctree = treesummary.max_clade_cred_tree()
			weight = 1.0
			treecount = len(chainb)-BURNIN
			mcctree = my_set_ca_node_depths(mcctree, [weight, treecount, 0, chaina[BURNIN:]])
			#print(mcctree)
			#print(mcctree[0].newick())
			### dist in phylolib ###### 
			"""
			rf = tree1.treedist_RF(tree2)
rfnorm = tree1.treedist_RF(tree2, normalise=True)
rfsimnorm = 1 - rfnorm
pd = tree1.treedist_pathdiff(tree2)
print(f"Robinson-Foulds distance: {rf}")
print(f"Normalised similarity (based on RF distance): {rfsimnorm:.2f}")
print(f"Path difference distance: {pd:.2f}")
			
			"""
			init_tree = Tree(1)
			init_tree.str2tree(chaina[0],20,by='io')
			init_tree.obs_time = init_tree.head.find_max_dist()
			

			# t2.convert_to_ultrametric()

			
			ml_tree = Tree(1)
			ml_tree.str2tree(chaina[BURNIN:][ind],20,by='nw')
			ml_tree.obs_time = ml_tree.head.find_max_dist()
			
			mcc_tree = Tree(1)
			mcc_tree.str2tree(mcctree[0].newick(),20,by='nw')
			mcc_tree.head.time =  ml_tree.obs_time - mcc_tree.head.right.find_max_dist()
			mcc_tree.obs_time = mcc_tree.head.find_max_dist()
			
			lst_lbd = chaind[BURNIN:] 
			ind_lbd = np.argmax(lst_lbd)
			s_lbd = chaina[BURNIN:][ind_lbd]
			lbd_tree = Tree(1)
			lbd_tree.str2tree(chaina[BURNIN:][ind_lbd],20,by='nw')
			lbd_tree.obs_time = lbd_tree.head.find_max_dist()
			
			# print(mcc_tree.obs_time, lbd_tree.obs_time)
			# t2.convert_to_ultrametric()

			
			if init_tree.obs_time + TOLER <  ml_tree.obs_time or init_tree.obs_time - TOLER > ml_tree.obs_time:
				print("\t FALSE ", init_tree.obs_time, ml_tree.obs_time )
			else:
				print("\t TRUE ", init_tree.obs_time, ml_tree.obs_time )
				alpha, d, D0, B = calc_param_means(chainc) # remove burn in
				N = len(chainc[BURNIN:]) 
				# find_MBT_stats(chainb, i)
				print("\talpha: ", sum(alpha[BURNIN:])/N)
				print("\td: ", sum(d[BURNIN:])/N)
				print("\tD0: ", sum(D0[BURNIN:])/N)
				print("\tB: ", sum(B[BURNIN:])/N)


						
						
				print("\t ml alpha: ", alpha[BURNIN:][ind])
				print("\t ml d: ", d[BURNIN:][ind])
				print("\t ml D0: ", D0[BURNIN:][ind])
				print("\t ml B: ", B[BURNIN:][ind])
				rf_totals_map, rf_totals_lbd, rf_totals_mcc = plot_stats(init_tree, chaina[BURNIN:][ind], s_lbd, mcc_tree, chaina, ind, ind_lbd, rf_totals_map, rf_totals_lbd, rf_totals_mcc, alpha, d, D0, B)
				# rf_lbd = MLE_lbd(chaina[BURNIN:], chainb[BURNIN:], chainc[BURNIN:], alpha, d, D0, B, rf_lbd)
				
				
	plotfig(np.array(rf_totals), "normalised rf distance", "normalised RF distance", clear=True)
	#plotfig(np.array(rf_lbd), "normalised rf distance with LBD prior", clear=True)
	#print("sample size = ", len(rf_totals))

	#print(find_pop_curve(t_cur))




