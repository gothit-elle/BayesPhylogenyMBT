import numpy as np
from nodestruct import *
from treestruct import *

import random

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from MCMCMoves import *
import csv
from scipy.linalg import expm

BASES = ["A","C", "G", "T"]
MAPPING_2P = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}

def sim_evo(cur, Q, Pi, seq_len):
	if cur == None:
		return None
	for seq_index in range(seq_len): # simulate a whole sequence
		# if cur.seq == 'N':
		if cur.seq == 'Failed':
			return cur
		u = np.random.uniform(0,1)
		# select state

		if cur.parent == None: # at the root so use the Pi vector for this
			sum = 0
			state = 0
			# this is wrong
			sum = Pi[0]
			state= 0
			# print('alpha is:', alpha, u)
			for i in range(len(Pi[1:])):
				elem = Pi[i+1]
				# print(sum, u, sum+elem)
				if (sum < u) and (u < (sum + elem)):
					state = i+1
				else:
					sum += elem
		else: # the state is determined by the parents end state
			state = BASES.index(cur.parent.seq[seq_index])
			
		# we have our state so now lets have a seq 
		u = np.random.uniform(0,1)
		res = event(expm(Q*cur.time), u, state, False, 0) 
		state = int(res[1])
		if cur.seq == None or cur.seq == 'N':
			cur.seq = BASES[state]
		else:
			cur.seq += BASES[state]

	if not cur.isLeaf():
		cur.right.parent = cur
		cur.left.parent = cur
	cur.right = sim_evo(cur.right, Q, Pi, seq_len)
	cur.left = sim_evo(cur.left, Q, Pi, seq_len)
	return cur
	
def event(mtrx, u, state, skip, sum):
	# print('mtrx is', mtrx[state,:], u)
	for i in range(len(mtrx[state, :])):
		elem = mtrx[state, i]
		if skip and i == state: # ignore diagonals
			pass
		elif sum < u and u < sum + elem:
			# print('new trans state is', i)
			return (True, i)
		else:
			sum += elem
	return (False, sum)

def event_d(mtrx, u, state, sum):
	# print('mtrx d is', mtrx[state], u, sum)
	if sum < u and u < mtrx[state] + sum:
		return (True, state)
	else:
		sum += mtrx[state]
	return (False, sum)
	
	
def sim_MBT(alpha, D0, d, B, cur, time, stopping_time):
	# print("func called", time)

	cur.seq = 'N'
	u = np.random.uniform(0,1)
	# select state
	sum = alpha[0]
	state= 0
	# print('alpha is:', alpha, u)
	for i in range(len(alpha[1:])):
		elem = alpha[i+1]
		# print(sum, u, sum+elem)
		if (sum < u) and (u < (sum + elem)):
			state = i+1
		else:
			sum += elem
	# print('state is', state)
	# so now we are in state i
	
	# need the 'loss' out of state i
	loss = -D0[state,state]
	
	t = np.random.exponential(loss) # an event happens
	cur.time += t
	cur.time = float(cur.time)
	if time + t> stopping_time: # force stop
		# print("reached obs time", time, t)
		cur.time = stopping_time - time
		return cur

	# the branch keeps living until it dies. so we dont care about length
	
	u = np.random.uniform(0,1)
	
	res = event(D0/loss, u, state, True, 0)
	if res[0] == True: # a transition occurred
		# print("t occ")
		state = res[1] # now exist in this state
		state_new = [0]*len(alpha)
		state_new[state] = 1
		cur = sim_MBT(state_new, D0, d, B, cur, time+t, stopping_time)
		
	elif res[0] == False: # no transition occurred. try a different event
		res = event_d(d/loss, u, state, res[1])
		if res[0] == True: # a death occurred
			# print("d occ")
			cur.seq = 'F'
			return cur # return as is
		elif res[0] == False:
			res = event(B/loss, u, state, False, res[1])
			
			states = MAPPING_2P[res[1]]
			state_l = [0]*len(alpha)
			# child phase
			state_l[states[0]] = 1
			
			state_r = [0]*len(alpha)
			# new parent phase
			state_r[states[1]] = 1
			# print("b occ")
			# a birth must have occurred.
			# get new states
			cur.left = node('N', cur, 0)
			cur.right = node('N', cur, 0)
			cur.left = sim_MBT(state_l, D0, d, B, cur.left, time+t, stopping_time)
			cur.right = sim_MBT(state_r, D0, d, B, cur.right, time+t, stopping_time)
	return cur

def merge_tree(cur):
	if cur.isLeaf():
		return cur
	if cur.right == None: # merge cur and left child
		cur.time += cur.left.time
		cur.map = cur.left.map
		cur.seq = cur.left.seq
		cur.right = cur.left.right
		cur.left = cur.left.left
		if cur.right != None: cur.right.parent = cur
		if cur.left != None: cur.left.parent = cur

	elif cur.left == None: # merge cur and right child
		cur.time += cur.right.time
		cur.map = cur.right.map
		cur.seq = cur.right.seq
		cur.left = cur.right.left
		cur.right = cur.right.right
		if cur.right != None: cur.right.parent = cur
		if cur.left != None: cur.left.parent = cur
	return cur
		
def clean_tree(cur):
	if cur == None:
		return None
	cur.right = clean_tree(cur.right)
	cur.left = clean_tree(cur.left)
	cur = merge_tree(cur)
	cur.fix_parents()
	return cur
		

def sim_tree(alpha, D0, d, B, Q1, Pi, time, min_leaves = 2, seq_len = 1, debug = False):
	t2 = Tree(1)
	t2.head = node('N', None, 0)
	t2.head = sim_MBT(alpha, D0, np.array(d), B, t2.head, 0, time)
	t2.head.prune_tree()
	t2.head = clean_tree(t2.head)
	while t2.head.right == None or t2.head.left == None or len(t2.head.find_leaves()) < min_leaves:
		t2.head = sim_MBT(alpha, D0, np.array(d), B, t2.head, 0, time)
		t2.head.prune_tree()
		t2.head = clean_tree(t2.head)



	t2.obs_time = t2.head.find_max_dist()
	t2.seq_len = seq_len
	t2.head = sim_evo(t2.head, Q1, Pi, seq_len)
	t2.head.map_leaves()

	# need to grow all leaves to max leaf dist.
	t2.head.alter_leaves(t2.obs_time)
	return t2

def target(s, N, t, Q1, alpha, d, D0, B, Pi, i, pos, multip):
	with open(parentdir + '/thesis_likelihood/logs/logr.txt', "w", encoding="utf-8") as f:
		successes, chaina, chainb, chainc = run_chain(s, N, t, Q1, alpha, d, D0, B, Pi, by='io', fname=f, pos=pos, send_tree=False, multip=multip)

	print("acceptance rate", successes/len(chaina))
	with open(parentdir + f"/csv/c{i+1}a.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chaina)

	with open(parentdir + f"/csv/c{i+1}b.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chainb)
		
	with open(parentdir + f"/csv/c{i+1}c.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chainc)
		

