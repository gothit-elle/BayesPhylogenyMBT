""" tree class
"""
import re
from nodestruct import *

class Tree:
  def __init__(self, seq_len = 0):
    self.head = None
    self.lik = None
    self.seq_len = seq_len
    self.obs_time = None

  def disp(self, log=False, fname=None):
    print(self.head, file = fname)

  def toStr(self):
    #convert tree to string
    return self.head.toStr()

  def fix_leaves(self):
    leaves = self.head.find_leaves()
    dists = self.head.find_leaf_dists()
    for i in range(len(dists)):
      dist = dists[i]
      val = self.obs_time - dist[0]
      tolerance = dist[1]
      # print(leaves[i].seq, val)
      if abs(val):
        # mismatch, needs rescaling
        if val > 0: # we are slightly under. 'grow' the branch
          # print("case a")
          leaves[i].time += val
        elif val < 0 and abs(val) < tolerance: # we have a branch of length tolerance, and our value is val units under (if its over it doesnt matter)
          # print("case b")
          leaves[i].time += val
        else: # damn how did this happen? need to rescale the entire tree.
          max_dist = self.head.find_max_dist()
          self.head.alter_leaves(max_dist) # grow the leaves
          scale_factor = self.obs_time/max_dist
          self.head.scale_tree(scale_factor)
          dists = self.head.find_leaf_dists() # update dists
          # print("dists updated", dists)

  def fix_tree(self):
    self.head.fix_parents()
    self.head.parent = None
    self.fix_leaves()


  def str2tree(self, nodeStr, t=None, by="df", debug=False, tol = 0.001):
    if by == 'io':
      # convert string to tree (in order traversal)
      seq = re.search("[A-Z]+",nodeStr).group()
      time= re.search("\d+[.]?\d*",nodeStr).group()
      #if "N" in seq:
      #  seq = None
      new_node = node(seq, None, float(time))
      cur = new_node
      self.seq_len = len(seq)
      skip = self.seq_len+len(time)
      for i in range(skip,len(nodeStr)):
        elem = nodeStr[i]
        if elem == "(": # create child node
          time= float(re.search("\d+[.]?\d*",nodeStr[i:]).group())
          seq = re.search("[A-Z]+",nodeStr[i:]).group()
          if len(seq) > self.seq_len:
            self.seq_len = len(seq)
          mapping = re.search("[A-Z]+[^\d]*(?!='\(|'\))",nodeStr[i:]).group().strip("'()").split("' + '")
          if len(mapping) == 2: 
            map = mapping[1]
          else:
            map = None
          # if "N" in seq:
          #  seq = None
          if type(cur.left) != type(node(None, None, 0)):
            cur.left = node(seq, cur, time)
            cur.left.map = map
            cur = cur.left
          else:
            cur.right = node(seq, cur, time)
            cur.right.map = map
            cur = cur.right
        if elem == ")":
          cur = cur.parent
    elif by=='df':
      nodeStr.upper()
      if nodeStr[-1] == '.':
        nodeStr = nodeStr[:-1]
      elems = nodeStr.split('.')
      # print(elems)

      time= float(re.search("\d+[.]?\d*",elems[0]).group())
      seq = re.search("[A-Z]+",elems[0]).group()
      #if "N" in seq:
      #  seq = None
      new_node = node(seq, None, time)
      cur = new_node
      self.seq_len = len(seq)
      time_tracker = time

      for i in range(len(elems[1:])):
        elem = elems[i+1]
        # we have a new element.

        time= float(re.search("\d+[.]?\d*",elem).group())
        seq = re.search("[A-Z]+",elem).group()
        # if "N" in seq:
        #    seq = None
        # print(time, seq)

        # print("timetracker is", time_tracker)
        # now check if the time of the child we want to add will take it beyond whats possible
        if time_tracker + time > t: # impossible, so cant add a child
          while (time_tracker + time > t):
            # print("at", cur.seq, cur.time)
            time_tracker -= cur.time
            cur=cur.parent

        time_tracker += time
        # if the left child is not already filled
        if type(cur.left) != type(node(None, None, 0)):
          # print("placing on the left:", seq, time)
          cur.left = node(seq, cur, time)
          cur = cur.left

        # the right child is not already filled
        elif type(cur.right) != type(node(None, None, 0)):
          # print("placing on the right:", seq, time)
          cur.right = node(seq, cur, time)
          cur = cur.right
        # already has 2 children
        else:
          time_tracker -= cur.time
          cur = cur.parent

    self.head = new_node
    self.obs_time = self.head.dist_from_tip() + self.head.time

# end struct