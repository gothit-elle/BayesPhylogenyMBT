from treestruct import *
from nodestruct import *
from itertools import combinations 
import numpy as np

def root_dist_inc(cur):
	steps = 0
	dist = cur.time
	while cur.parent != None:
		steps += 1 
		dist += cur.parent.time
		cur = cur.parent

	return dist, steps # make a note that we add the root distance. this changes the metric slightly
	
	
def kc_metric(tree, weight=0.5):
	step_arr = []
	dist_arr = []
	pendant_arr = []
	leaves, parents = find_levels(tree)
	pairs = list(combinations(leaves, 2))
	for pair in pairs:
		leaf1, leaf2 = pair
		cur_leaf = leaf1
		current_node_set = [leaf1]
		cur_parent = cur_leaf.parent
		
		while leaf2 not in current_node_set:
			if cur_leaf == cur_parent.right:
				# right child, so look at left
				current_node_set += cur_parent.left.find_leaves()
			else:
				current_node_set += cur_parent.right.find_leaves()
			if leaf2 in current_node_set:
				break
			temp = cur_parent 
			cur_parent = cur_parent.parent
			cur_leaf = temp
		mrca = cur_parent
		Mij, mij = root_dist_inc(mrca)
		step_arr.append(mij)
		dist_arr.append(Mij)
	for leaf in leaves:
		pendant_arr.append(leaf.time)
	step_arr += [1]*len(pendant_arr)
	dist_arr += pendant_arr

	return (1-weight)*np.array(step_arr) + weight*np.array(dist_arr)
	

def kc_dist(t1, t2, weight=0.5):
	v1 = kc_metric(t1, weight)
	v2 = kc_metric(t2, weight)
	return np.linalg.norm(v1-v2)
	
	
	
"""
t1 = Tree(1)
t1.head = node("Root", None, 0.1)
t1.head.right = node("N", t1.head, 1.1)
t1.head.left = node("N", t1.head, 0.5)
t1.head.left.right = node("B", t1.head.left, 0.8)
t1.head.left.left = node("A", t1.head.left, 1.2)
t1.head.right.right = node("D", t1.head.left, 1)
t1.head.right.left = node("C", t1.head.left, 0.8)

t1.head.map = "root"
t1.head.right.map = "N"
t1.head.left.map = "N"
t1.head.left.right.map = "B"
t1.head.left.left.map = "A"
t1.head.right.right.map = "D"
t1.head.right.left.map = "C"
t1.disp()

t2 = Tree(1)
t2.head = node("Root", None, 0.1)
t2.head.right = node("N", t2.head, 1.1)
t2.head.left = node("N", t2.head, 0.5)
t2.head.left.right = node("B", t2.head.left, 0.8)
t2.head.left.left = node("A", t2.head.left, 1.2)
t2.head.left.right.right = node("D", t2.head.left, 1)
t2.head.left.right.left = node("C", t2.head.left, 0.8)

t2.head.map = "root"
t2.head.right.map = "N"
t2.head.left.map = "N"
t2.head.left.right.map = "B"
t2.head.left.left.map = "A"
t2.head.left.right.right.map = "D"
t2.head.left.right.left.map = "C"
t2.disp()

print(kc_dist(t1,t2))
"""