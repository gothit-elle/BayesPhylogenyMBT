import numpy as np
from nodestruct import *
from treestruct import *

import random
import sys
from MCMCMoves import *
import csv

BASES = ["A","C", "G", "T"]
MAPPING_2P = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}

def sim_evo(cur, Q, Pi):

	if cur == None:
		return None
	# if cur.seq == 'N':
	if cur.seq == 'Failed':
		return cur
	u = np.random.uniform(0,1)
	# select state
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
	
	total_time = 0 
	while total_time < cur.time:
		cur.seq = BASES[state]
		# print(cur.seq)
		# state = BASES.index(cur.seq)
		loss = -Q[state, state]
		t = -1/loss * np.log(u)
		total_time += t
		if total_time > cur.time: # the next event will only happen after the branch ends, so ignore it
			break
		u = np.random.uniform(0,1)
		res = event(Q/loss, u, state, True, 0) 
		state = res[1]
	cur.seq = BASES[state]
	
	pi_new = [0]*len(Pi)
	pi_new[state] = 1
	cur.right = sim_evo(cur.right, Q, pi_new)
	cur.left = sim_evo(cur.left, Q, pi_new)
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
	
	u = np.random.uniform(0,1)
	t = -1/loss * np.log(u) # an event happens
	cur.time += t
	if time + t> stopping_time: # force stop
		# print("reached obs time", time, t)
		cur.time = stopping_time - cur.dist_from_root()
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
	if cur.right == None:
		cur.left.time += cur.time
		cur = cur.left
	elif cur.left == None:
		cur.right.time += cur.time
		cur = cur.right 
	return cur
		
def clean_tree(cur):
	if cur == None:
		return None
	cur.right = clean_tree(cur.right)
	cur.left = clean_tree(cur.left)
	cur = merge_tree(cur)
	return cur
		

def sim_tree(alpha, D0, d, B, Q1, Pi, time, debug = False):
	t2 = Tree(1)
	t2.head = node('N', None, 0)
	t2.head = sim_MBT(alpha, D0, np.array(d), B, t2.head, 0, time)
	if debug: t2.disp()
	t2.head.prune_tree()
	if debug: t2.disp()
	t2.head = clean_tree(t2.head)
	if debug: t2.disp()
	t2.head = sim_evo(t2.head, Q1, Pi)
	t2.head.map_leaves()
	return t2

def target(s, N, t, Q1, alpha, d, D0, B, Pi, i, pos):
	with open('../thesis_likelihood\logs\logr.txt', "w", encoding="utf-8") as f:
		successes, chaina, chainb, chainc = run_chain(s, N, t, Q1, alpha, d, D0, B, Pi, by='io', fname=f, pos=pos)

	print("acceptance rate", successes/len(chaina))
	with open(f"../thesis_likelihood\csv\c{i+1}a.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chaina)

	with open(f"../thesis_likelihood\csv\c{i+1}b.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chainb)
		
	with open(f"../thesis_likelihood\csv\c{i+1}c.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chainc)
		

