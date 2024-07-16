from bs4 import BeautifulSoup
import sys
sys.path.insert(0, '../thesis_likelihood')
from nodestruct import *
from treestruct import *
import re
from math import comb
import numpy as np

with open('../thesis_likelihood\data\small_dataset.xml', 'r') as f:
	data = f.read()
Bs_data = BeautifulSoup(data, 'xml')

seqs = Bs_data.find_all('sequence')
# taxa= seqs = Bs_data.find_all('taxa')

seq_dict = {}
for seq in seqs:
	txt = seqs[0].text
	txt = txt.replace('\n', '')
	txt = txt.replace('\t', '')
	txt = txt.replace('-', 'N')
	seq_dict[seq.taxon.get('idref')] = txt
print(list(seq_dict.items())[0:2])



def two(x):
	a = iter(x)
	return zip(a,a)

def make_leaves(leaves, time):
	t = np.random.exponential(time) #comb(len(leaves), 2)
	leaf_nodes = []
	for key, seq in leaves:
		new_node = node(seq, None, t)
		new_node.map = key
		leaf_nodes.append(new_node)
	return leaf_nodes
	
def tree_from_leaves(leaves, time):
	parents = []
	time = np.random.exponential(time) #comb(len(leaves), 2)
	for l1, l2 in two(leaves):
		parent = node('N', None, time)
		l1.parent = parent
		l2.parent = parent
		parent.left = l1
		parent.right = l2
		parents.append(parent)
	if len(leaves) % 2 == 1: # odd number of joins
		l1 = parents[-1]
		l2 = leaves[-1]
		parent = node('N', None, time)
		l1.parent = parent
		l2.parent = parent
		parent.left = l1
		parent.right = l2
		parents[-1] = parent
	if len(parents) > 1:
		parents = tree_from_leaves(parents, time)
		
	return parents

#test = [(None, 'A'), ('TB', 'B'), ('TC', 'C'), ('TD', 'D'), ('TE', 'E')]
test = list(seq_dict.items())

def root_tree(obs, time):
	leaves = make_leaves(obs, time)
	root = tree_from_leaves(leaves, time)
	t_cur = Tree(1)
	t_cur.head = root[0]
	t_cur.seq_len = len(obs[0])
	t_cur.obs_time = t_cur.head.time + t_cur.head.find_max_dist()
	t_cur.fix_tree()
	return t_cur

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
		
		
t2 = root_tree(test, 3)
t2.disp()
str = t2.toStr()
print(str)
tree_new = Tree(1)
tree_new.str2tree(str, 5, by='io')
tree_new.disp()