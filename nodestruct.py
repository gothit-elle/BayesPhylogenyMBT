from num2words import num2words
BASES = ["A","C", "G", "T"]
# Tree data struct

class node:
  def __init__(self, seq, parent, time):
    self.seq = seq
    self.right = None
    self.left = None
    self.parent = parent
    self.time = time
    self.lik = None
    self.map = None
    self.prior = None
    self.gval = None
    self.changed = False

  def __str__(self, level=0):
    #ret = "\t"*level+"time: " + repr(self.time)+' / '+repr(self.seq)+"\n" # original
    if self.map != None:
      mid_val = self.map
    else:
      mid_val = 'N' #self.seq
    ret = repr(round(self.time,2)) + ' ~ ' + repr(mid_val)+ ' ~ ' +repr(round(self.dist_from_tip(),2)) +"\n"
    if level != 0:
      ex = ' '*round((level+1)*1.8 + 0.9)
    else:
      ex = ''
    if (self.right is not None):
      ret += "\t"*(level+1) + ex + "├──" + self.right.__str__(level+1)
    if (self.left is not None):
      ret += "\t"*(level+1)+ ex +"└──" + self.left.__str__(level+1)
    return ret

  def __repr__(self):
    return '<tree rep>'
    
  def is_neg(self):
    if self == None:
      return False
    if self.time <= 0 or (self.right != None and self.right.is_neg()) or (self.left != None and self.left.is_neg()):
      return True
      
  def isLeaf(self):
    if (self.left == None) and (self.right==None):
      return True

  def dist_from_tip(self):
    t = 0
    if self.isLeaf():
      return 0
    cur = self
    while cur != None:

      if cur.right != None:
        cur=cur.right
      elif cur.left != None:
        cur=cur.left
      else:
        return t
      t += cur.time

  def associative_mark(self):
    if self.changed == 1 or self.changed == -1:
      return -1
    else:
      ret = 0
      if self.right != None: ret += self.right.associative_mark()
      if self.left != None: ret += self.left.associative_mark()
      self.changed = max(-1, ret)
      return self.changed
	 
  def mark_tree(self):
    self.changed = True
    if self.right != None: self.right.mark_tree()
    if self.left != None: self.left.mark_tree()

  def scale_tree(self, delta):
    self.time *= delta
    self.changed = True
    if self.right != None:
      self.right.scale_tree(delta)
    if self.left != None:
      self.left.scale_tree(delta)

  def alter_leaves(self, scale):
    if self.isLeaf():
      self.time = scale
      self.changed = True
    else:
      self.right.alter_leaves(scale-self.time)
      self.left.alter_leaves(scale-self.time)

  def dist_from_root(self):
    dist = 0
    cur = self
    while cur.parent != None:
      dist += cur.parent.time
      cur = cur.parent
    return dist

  def scale_parents(self, delta, obs_time):
    if self.isLeaf():
      dist = self.dist_from_root()
      self.time = obs_time - dist
      self.changed = True
    else:
      self.time *= delta
      self.changed = True
      self.right.scale_parents(delta, obs_time)
      self.left.scale_parents(delta, obs_time)

  def can_prune(self):
    for leaf in self.find_leaves():
      if leaf.seq == 'F':
        return True
    return False
    
  def prune_tree(self):
    leaves= self.find_leaves()
    while self.can_prune():
      for leaf in leaves: 
        if leaf.seq == 'F':
          if leaf.parent == None:
            leaf.seq = "Failed"
          elif leaf.parent.right == leaf:
            leaf.parent.right = None
          else:
            leaf.parent.left = None

            
  def find_max_dist(self):
    d1= 0
    d2= 0
    if self == None:
      return 0
    if self.right != None:
      d1 = self.right.find_max_dist()
    if self.left != None:
      d2 = self.left.find_max_dist()
    max_dist = self.time + max(d1, d2)
    return max_dist

  def find_min_dist(self):
    d1= 0
    d2= 0
    if self == None:
      return 0
    if self.right != None:
      d1 = self.right.find_min_dist()
    if self.left != None:
      d2 = self.left.find_min_dist()
    min_dist = self.time + min(d1, d2)
    return min_dist

  def base2int(self,index):
    if self.seq[index] == "N":
      return -1
    return [int(self.seq[index]==char) for char in BASES]

  def toStr(self):
    # convert subtree to string
    ret = ""
    if self is not None:
      ret += "("
      #ret += repr(self.time)+' / '+repr(self.seq)
      ret += repr(self.time)+' / '+repr(self.seq)
      if self.map is not None:
        ret += ' + ' + repr(self.map)
      if not self.isLeaf():
        ret += self.left.toStr()
      if not self.isLeaf():
        ret += self.right.toStr()
      ret += ")"
    return ret
    
    
# two functions below copied from https://stackoverflow.com/a/61123048


  def traverse(self, newick):
    
    if self.left and not self.right:
      map = self.map
      newick = f"(,{self.left.traverse(newick)}){map}:{self.time}"
    elif not self.left and self.right:
      map = self.map
      newick = f"({self.right.traverse(newick)},){map}:{self.time}"
    elif self.left and self.right:
      map = self.map
      newick = f"({self.right.traverse(newick)},{self.left.traverse(newick)}){map}:{self.time}"
    elif not self.left and not self.right:
      map = self.map
      newick = f"{map}:{self.time}"
    else:
      pass
    return newick
    
  def to_newick(self):
    newick = ""
    newick = self.traverse(newick)
    newick = f"{newick};"
    return newick

  def fix_parents(self):
    if self.right != None:
      self.right.parent = self
      self.right.fix_parents()
    if self.left != None:
      self.left.parent = self
      self.left.fix_parents()

  def find_leaves(self):
    cur = self
    leaves = []
    if cur.isLeaf():
      return [cur]
    l_r = []
    if cur.right != None:
      l_r = cur.right.find_leaves()
    l_l = []
    if cur.left != None:
      l_l = cur.left.find_leaves()
    leaves += l_r + l_l
    return leaves

  def find_parents(self):
    parents = []
    if not self.isLeaf():
      parents += [self]
      parents += self.right.find_parents()
      parents += self.left.find_parents()
    return parents
    
  def find_leaf_dists(self):
    dists = []
    leaves = self.find_leaves()
    for leaf in leaves:
      cur = leaf
      dist = 0
      while(cur!=None):
        dist += cur.time
        cur = cur.parent
      dists.append((dist, leaf.time))
    return dists

  def map_leaves(self):
    leaves = self.find_leaves()
    for i in range(len(leaves)):
      leaf = leaves[i]
      leaf.map = f"Species{num2words(i).title()}"


      