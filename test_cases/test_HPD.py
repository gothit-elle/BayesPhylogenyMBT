import csv
import matplotlib.pyplot as plt
import os
import sys
import inspect


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
from kendall_colijn import *
from prior import *
from scipy.linalg import expm

TOLER = 5
BURNIN = 5000
look_in = gparentdir + "/thesis_likelihood_csv_files/csv/"
LIMIT = 30
STEPS = 100
#look_in = parentdir + "/csv/"

def set_ca_node_depths(sum_tree, wt, count, burnin, filename,):
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

	ntrees = count - burnin
	wsum += weight
	multiplier = weight / ntrees
	treefile = Treefile(filename)
	for i in range(burnin):
		treefile.readtree(returntree=False)
	for input_tree in treefile:
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
	ax = ptype(x1) #,stat='count',  binwidth=0.05,
	params=fit.fit(x1)
	#print(params)
	#xx = np.linspace(*ax.get_xlim(),100)
	#ax.plot(f(xx), f(fit.pdf(xx, *params)), color = next(palette) )
	
	if clear: 
		ax.set_title(f"{pname} boxplot")
		ax.figure.savefig(parentdir + f"/plots/{name}.png", bbox_inches='tight')
		ax.figure.clf()
	
def plot_arr(array, hash, name):

	#N = len(array)
	#plotfig(pd.DataFrame(array), f"{hash}_{name}",name, clear = True)
	pass
		

def TTE(alpha, d, D0, B, x, adjust=False):
	alpha = alpha.astype(object)
	D0 = D0.astype(object)
	B = B.astype(object)
	d = d.astype(object)
	if adjust: 
		gval = G_bkxk(1,x-1, alpha, d, D0.reshape(2,2), B.reshape(2,4))
		alpha = alpha@gval
	res = get_E(LIMIT, alpha, d, D0.reshape(2,2), B.reshape(2,4), plot=1).y
	return alpha@res


def MTPS(time, alpha, d, D0, B, x, adjust=False):
	alpha = alpha.astype(object)
	D0 = D0.astype(object)
	B = B.astype(object)

	if adjust: 
		gval = G_bkxk(1,x-1, alpha, d, D0.reshape(2,2), B.reshape(2,4))
		alpha = alpha@gval
		print("alpha is", alpha)
	times = np.linspace(0,time)
	N = len(B.reshape(2,4))
	prod = np.transpose(np.kron(np.ones(N),np.identity(N)) + np.kron(np.identity(N),np.ones(N))) # this transpose is once again sus

	omega = D0.reshape(2,2) + B.reshape(2,4)@(prod)
	print("omega is", omega)

	return [alpha@expm(omega*t)@np.ones(N) for t in times]
		

	
def TTE_LBD(alpha, d, D0, B, x, adjust=False):
	alpha = 1
	if adjust:
		lambda0 = np.average(B.astype(object))
	else:
		lambda0 = np.average(B.astype(object).reshape(2,4)@np.ones(4))
	mu0 = np.average(d.astype(object))
	
	times = np.linspace(0, LIMIT, 100)
	def elbd(lambda0, mu0, t):
		return 1-(lambda0-mu0)/(lambda0-mu0*np.exp(-(lambda0-mu0)*t))
	res = [elbd(lambda0, mu0, t) for t in times]
	return res


def MTPS_LBD(time, alpha, d, D0, B, x, adjust=False):
	alpha = 1
	if adjust:
		B = np.average(B.astype(object))
	else:
		B = np.average(B.astype(object).reshape(2,4)@np.ones(4))
	d = np.average(d.astype(object))
	D0 = -B - d

	times = np.linspace(0,time)

	prod = 2 

	omega = D0+ B*(prod)
	print("omega is", omega)

	return [np.exp(omega*t) for t in times]
	

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
	"""
	mlpc = find_pop_curve(mlt_str, by='nw')
	tpc = find_pop_curve(chaina[0], by='io')
	lbdpc = find_pop_curve(lbdt_str, by='nw')
	"""
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

	"""
	pool = Pool()
	results = pool.starmap(plot_arr, zip([alpha, d, D0, B], repeat(i), ["alpha", "d", "D0", "B"])) 
	pool.close()
	"""
	"""
	pool = Pool()
	results = pool.map(find_pop_curve, chaina[BURNIN:]) # remove burn in
	pool.close()
	times = np.linspace(0, init_tree.obs_time, 50)
	mean_pops = sum(results)/len(results)
	"""
	
	mean_total_pop_true = MTPS(init_tree.obs_time, alpha[0],  d[0], D0[0], B[0], init_tree.obs_time)
	extinction_time_true = TTE(alpha[0], d[0], D0[0], B[0], init_tree.obs_time)
	
	mean_total_pop_MAP = MTPS(init_tree.obs_time, alpha[BURNIN:][ind_mle], d[BURNIN:][ind_mle], D0[BURNIN:][ind_mle], B[BURNIN:][ind_mle], init_tree.obs_time)
	extinction_time_MAP = TTE(alpha[BURNIN:][ind_mle], d[BURNIN:][ind_mle], D0[BURNIN:][ind_mle], B[BURNIN:][ind_mle], init_tree.obs_time)
	
	mean_total_pop_LBD = MTPS_LBD(init_tree.obs_time, alpha[BURNIN:][ind_lbd], d[BURNIN:][ind_lbd], D0[BURNIN:][ind_lbd], B[BURNIN:][ind_lbd], init_tree.obs_time, adjust=False)
	extinction_time_LBD = TTE_LBD(alpha[BURNIN:][ind_lbd], d[BURNIN:][ind_lbd], D0[BURNIN:][ind_lbd], B[BURNIN:][ind_lbd], init_tree.obs_time, adjust=False)
		
	#mean_total_pop_LBD = MTPS(alpha[BURNIN:][ind_LBD], D0[BURNIN:][ind_LBD], B[BURNIN:][ind_LBD].reshape(2,4))
	#extinction_time_LBD = TTE(alpha[BURNIN:][ind_LBD], d[BURNIN:][ind_LBD], D0[BURNIN:][ind_LBD], B[BURNIN:][ind_LBD].reshape(2,4))
	

	# plt.plot(times, mccpc, label="MCC pop curve")
	"""
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
	plt.savefig(f"../thesis_likelihood/plots/meanpop_{i}.png", bbox_inches='tight')
	plt.clf()
	"""

	params = {'mathtext.default': 'regular' }
	plt.rcParams.update(params)
	plt.xlabel('t')
	plt.ylabel('M(t)')
	plt.title("Theoretical Mean Total Population Size")
	plt.plot(np.linspace(0,init_tree.obs_time), mean_total_pop_true, label="True")
	plt.plot(np.linspace(0,init_tree.obs_time), mean_total_pop_MAP, label="MAP MBT")
	plt.plot(np.linspace(0,init_tree.obs_time), mean_total_pop_LBD, label="MAP LBD")
	plt.legend()
	plt.savefig(f"../thesis_likelihood/plots/meanpop_theoretical_{i}.png", bbox_inches='tight')
	plt.clf()
	

	params = {'mathtext.default': 'regular' }
	plt.rcParams.update(params)
	plt.xlabel('t')
	plt.ylabel(r'$\alpha \mathbf{E}(t)$')
	plt.title("Theoretical Time to Extinction")
	plt.plot(np.linspace(0,LIMIT,STEPS), extinction_time_true, label="True")
	plt.plot(np.linspace(0,LIMIT,STEPS), extinction_time_MAP, label="MAP MBT")
	plt.plot(np.linspace(0,LIMIT,STEPS), extinction_time_LBD, label="MAP LBD")
	plt.legend()
	plt.savefig(f"../thesis_likelihood/plots/extinction_theoretical_{i}.png", bbox_inches='tight')
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
	rf_totals_map = []
	rf_totals_lbd = []
	rf_totals_mcc = []
	kc_map = []
	kc_lbd = []
	# Iterate directory

	for path in os.listdir(look_in):
		# check if current path is a file
		if os.path.isfile(os.path.join(look_in, path)):
			if path[-5] == 'a':
				hashes.append(path[0:-5])
	print(hashes)
	#i = 1 #"3929977c72a745b180364a64a6c3a74c"
	# i = "c056e72282e340e797642326d209c601" #this reqs skip rows
	
	for j in tqdm(range(len(hashes))):
		i = hashes[j]
		chaina, chainb, chainc, chaind = read_chains(i)
		
		
		
		
		if len(chainb) >= BURNIN:
			print("\nStep ", j, ", for dataset: ", i, "len", len(chainb))
			
			# check convergence rq
			N = len(chainb)
			"""
			plt.plot(range(N), chainb)
			params = {'mathtext.default': 'regular' }
			plt.rcParams.update(params)
			plt.xlabel('Step')
			plt.ylabel('Posterior Log-Likelihood')
			plt.title("Posterior Log-likelihood vs MCMC Step")

			plt.savefig(parentdir + f"/plots/MCMC_sim_{i}.png", bbox_inches='tight')
			plt.clf()
			"""
			lst = chainb[BURNIN:] 
			ind = np.argmax(lst)
			s = chaina[BURNIN:][ind]
			treesummary = pt.BigTreeSummary(trackbips=False, trackclades=True, trackroot=False)
			for str in chaina[BURNIN:]:
				str = format_str(str)
				mytree = pt.Tree.from_string(str)
				treesummary.add_tree(mytree)
				#with open(gparentdir + f"/thesis_likelihood_csv_files/treefiles/{i}.trees", "a") as outfile:
				#	outfile.write(mytree.nexus())
			#treefile = pt.Nexustreefile( gparentdir + f"/thesis_likelihood_csv_files/treefiles/{i}.trees")
			#for mytree in treefile:
				#treesummary.add_tree(mytree)
			mcctree = treesummary.max_clade_cred_tree()
			#weight = 1.0
			#treecount = len(chainb)-BURNIN
			#print(mcctree[0], [weight, treecount, 0, gparentdir + f"/thesis_likelihood_csv_files/treefiles/{i}.trees"])
			#mcctree = set_ca_node_depths(mcctree[0], weight, treecount, 0, gparentdir + f"/thesis_likelihood_csv_files/treefiles/{i}.trees")
			#print(mcctree)
			#print("mcctree is")
			#print(mcctree)
			### dist in phylolib ###### 

			init_tree = Tree(1)
			init_tree.str2tree(chaina[0],20,by='io')
			init_tree.obs_time = init_tree.head.find_max_dist()
			

			# t2.convert_to_ultrametric()

			
			ml_tree = Tree(1)
			ml_tree.str2tree(chaina[BURNIN:][ind],20,by='nw')
			ml_tree.obs_time = ml_tree.head.find_max_dist()
			#ml_tree.disp()
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
			
			kc_map.append(kc_dist(init_tree, ml_tree))
			kc_lbd.append(kc_dist(init_tree, lbd_tree))
			# print(mcc_tree.obs_time, lbd_tree.obs_time)
			# t2.convert_to_ultrametric()

			
			if init_tree.obs_time + TOLER <  ml_tree.obs_time or init_tree.obs_time - TOLER > ml_tree.obs_time:
				print("\t FALSE ", init_tree.obs_time, ml_tree.obs_time )
			else:
				print("\t TRUE ", init_tree.obs_time, ml_tree.obs_time )
				alpha, d, D0, B = calc_param_means(chainc) # remove burn in
				N = len(chainc[BURNIN:]) 
				# find_MBT_stats(chainb, i)
				print("\tinit alpha: ", alpha[0])
				print("\tinit d: ", d[0])
				print("\tinit D0: ", D0[0])
				print("\tinit B: ", B[0])


						
						
				print("\t ml alpha: ", alpha[BURNIN:][ind])
				print("\t ml d: ", d[BURNIN:][ind])
				print("\t ml D0: ", D0[BURNIN:][ind])
				print("\t ml B: ", B[BURNIN:][ind])
				rf_totals_map, rf_totals_lbd, rf_totals_mcc = plot_stats(init_tree, chaina[BURNIN:][ind], s_lbd, mcc_tree, chaina, ind, ind_lbd, rf_totals_map, rf_totals_lbd, rf_totals_mcc, alpha, d, D0, B)
				# rf_lbd = MLE_lbd(chaina[BURNIN:], chainb[BURNIN:], chainc[BURNIN:], alpha, d, D0, B, rf_lbd)
				
	data = {"MBT": rf_totals_map, "LBD": rf_totals_lbd, "MCC":rf_totals_mcc}
	plotfig(pd.DataFrame(data), "normalised rf distance", "normalised RF distance", clear=True)
	data2 = {"MBT": kc_map, "LBD": kc_lbd}
	plotfig(pd.DataFrame(data2), "Kendall-Colijn Distance", "Kendall-Colijn Distance", clear=True)
	
	data = {"MBT - LBD rf dist": [i-j for i,j in zip(rf_totals_map, rf_totals_lbd)], "MBT - MCC rf dist": [i-j for i,j in zip(rf_totals_map, rf_totals_mcc)]}
	plotfig(pd.DataFrame(data), "normalised rf distance difference", "normalised RF distance difference between the MBT and LBD", clear=True)
	data2 = {"MBT - LBD dist": [i-j for i,j in zip(kc_map, kc_lbd)]}
	plotfig(pd.DataFrame(data2), "Kendall-Colijn Distance difference", "Kendall-Colijn Distance difference between the MBT and LBD", clear=True)
	print("final data len was ", len(rf_totals_map))
	#plotfig(np.array(rf_lbd), "normalised rf distance with LBD prior", clear=True)
	#print("sample size = ", len(rf_totals))

	#print(find_pop_curve(t_cur))



