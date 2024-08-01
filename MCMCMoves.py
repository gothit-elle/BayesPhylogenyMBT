import numpy as np
from copy import deepcopy as dcpy
from treestruct import *
from posterior import *
from tqdm import tqdm
"""# MCMC"""

debug = 0

BETA = 1.2 # from paper [2]
theta = 1
theta = 1/theta
WINDOW_ALPHA = 0.1
WINDOW_D0 = 0.1
WINDOW_B = 0.1
WINDOW_MU = 0.1


def find_node_n(cur, call, pick_node):
    ret = None
    if type(call) != int:
      return call
    call += 1
    if cur == None:
      return call-1
    if pick_node == call:
      # print("we found it~!!", cur.seq, cur.time)
      ret = cur
    if ret == None:
      call = find_node_n(cur.left, call, pick_node)
      call = find_node_n(cur.right, call, pick_node)
      return call
    else:
      return ret

EXIT_FAILURE = -1

def propose_move(tree, alpha, d, D0, B, step, move_type=None, debug=False):
  Q = 0
  tcpy = dcpy(tree)
  alpha_cpy = dcpy(alpha)
  d_cpy = dcpy(d)
  D0_cpy = dcpy(D0)
  B_cpy = dcpy(B)
  cur = tcpy.head
  if move_type==None:
    move_type= np.random.choice(5)
  if debug:
    print(move_type)
    warnings.warn("Warning: theta set arbitrarily and not tied to parameters")
    warnings.warn("Warning: nodes are a fixed choice here")
  num_nodes = len(tcpy.toStr().split('(')) - 1
  
  if move_type == 0: # scaling move TODO: test
    delta = np.random.uniform(1/BETA, BETA)
    Q = 1/delta**(num_nodes-3) # this might need to be a -2 since we care about the scale of the root here

    if debug: print(delta,  tcpy.obs_time)
    tcpy.head.scale_parents(delta,  tcpy.obs_time) # scale all the internal nodes
    leaves = tcpy.head.find_leaves()
    for leaf in leaves:
      if leaf.time <=0:
        return EXIT_FAILURE


  elif move_type == 1: # WB move

    n_i = np.random.choice(num_nodes) + 1
    n_j = np.random.choice(num_nodes) + 1
    if debug:
      n_i = 7
      n_j = 5
    i = find_node_n(cur, 0, n_i)
    j = find_node_n(cur, 0, n_j)
    if debug: print("node i", n_i, i.seq, "node j", n_j, j.seq)
    replace_root = (j==1)


    ip = i.parent
    jp = j.parent
    
    if (jp != None and jp.dist_from_tip() <= i.dist_from_tip()) or (i == j) or (ip == j) or (ip == jp) or (ip == None):
      return EXIT_FAILURE
    else:
      ipp = i.parent.parent
      replace_root2 = (ipp == None)
      idft = i.dist_from_tip()
      jdft = j.dist_from_tip()
      
      ipdft = ip.dist_from_tip()


      if ip.left == i:
        k = ip.right
      else:
        k = ip.left


      # calculate Q
      if jp == None:
        delta = np.random.exponential(theta)
        Q = theta/(np.exp(-delta/theta)*(ipp.dist_from_tip()-max(i.dist_from_tip(), k.dist_from_tip())))
      else: # j is not the root
        #ip.time = np.random.uniform(np.max(i.time, j.time), jp.time)
        if ipp == None: # ip is the root
          #print(jp.dist_from_tip(), max(i.dist_from_tip(), j.dist_from_tip()), ip.dist_from_tip()-k.dist_from_tip())
          Q = (jp.dist_from_tip() - max(i.dist_from_tip(), j.dist_from_tip()))*np.exp(-(ip.dist_from_tip()-k.dist_from_tip())/theta)/theta
        else:
          #print("ipp", ipp.seq, "jp", jp.seq, "ijk", i.seq, j.seq, k.seq)
          #print("ipp", ipp.dist_from_tip(), "jp", jp.dist_from_tip(), "ijk", i.dist_from_tip(), j.dist_from_tip(), k.dist_from_tip())
          Q = (jp.dist_from_tip() - max(i.dist_from_tip(), j.dist_from_tip()))/(ipp.dist_from_tip()-max(i.dist_from_tip(), k.dist_from_tip()))

      if ipp != None:
        if ipp.left == ip:
          ipp.left = k
        else:
          ipp.right = k
      else:
        replace_root2 = True

      if jp != None:
        jpdft = jp.dist_from_tip()
        if debug: print("ip is", ip)
        if jp.left == j:
          jp.left = ip
        else:
          jp.right = ip
      else:
        replace_root=True
      if debug: print("jp is", jp.seq, jp.right.seq, jp.left.seq)
      if ip.left != i:
        ip.left = j
      else:
        ip.right = j
      if debug: print("ip is", ip.seq, ip.right.seq, ip.left.seq)
      if replace_root:
        tcpy.head = ip
      if replace_root2:
        if debug: print("k is", k.seq, k.right.seq, k.left.seq)
        if k.left == j:
          k.left = ip
        elif k.right == j: # idk about this 1
          k.right = ip
        tcpy.head = k


      if jp == None: # j is the root

        tcpy.head.time += delta
      else:
        new_dist = np.random.uniform(max(idft, jdft), jpdft)
        extend_branch_by = ipdft - new_dist
        ip.time += extend_branch_by

      if debug:
        print("tcpy is, before fixing:")
        tcpy.disp()
      tcpy.fix_tree() # we need to fix the tree.




  elif move_type==2: # Subtree Exchange
    n_i = np.random.choice(num_nodes) + 1
    # warnings.warn("Warning: nodes are a fixed choice here")
    # n_i = 3
    i = find_node_n(cur, 0, n_i)
    if debug: print("node i", n_i, i.seq, i.time, i.dist_from_tip())
    ip = i.parent
    if ip == None or ip.parent == None:
      return EXIT_FAILURE
    else:
      ipp = ip.parent

      if debug: print(ipp.seq)
      # k is the other child
      if ipp.left == ip:
        k = ipp.right
        ipp.right = i
      else:
        k = ipp.left
        ipp.left = i
      if ip.right == i:
        ip.right = k
      else:
        ip.left = k

    Q = 1
    tcpy.fix_tree() # we need to fix the tree

  elif move_type==3: # Node Age
    # finally the easiest move. why didnt i do this first.
    # this also has the most bugs. why

    # pick an internal node
    n_i = np.random.choice(num_nodes) + 1
    if debug: n_i = 1
    i = find_node_n(cur, 0, n_i)

    while i.isLeaf(): # we only want an internal node. try again
      n_i = np.random.choice(num_nodes) + 1
      i = find_node_n(cur, 0, n_i)

    ip = i.parent
    j = i.left
    k = i.right


    dj = float('inf')
    dk = float('inf')
    if j != None:
      dj = j.time
    if k != None:
      dk = k.time
    if j == None and k == None:
      return EXIT_FAILURE
    # if ip == None: #i is the root
    #   delta = np.random.uniform(1/BETA, BETA)
    #   # if debug: delta = 1/BETA
    #   range_dist = i.time + min(dj,dk)
    #   i.time = min(dj,dk) + delta*(i.time - delta*min(dj,dk))
    #   reduction = range_dist - i.time
    #   difference = abs(dk-dj)
    #   if dj < dk: # j is shorter
    #     j.time = reduction
    #     k.time = reduction + difference
    #     if k.time == float('inf'):
    #       return EXIT_FAILURE
    #   else:
    #     k.time = reduction
    #     j.time = reduction + difference
    #     if j.time == float('inf'):
    #       return EXIT_FAILURE
    # range_dist = i.time + min(dj,dk)
    # if i.time + delta*min(dj,dk) >= i.time + min(dj,dk):
    #   return EXIT_FAILURE
    # i.time += delta*min(dj,dk)
    # j.time -= delta*min(dj,dk)
    # k.time -= delta*min(dj,dk)
    # Q = 1/delta
    # else:
    range_dist = i.time + min(dj,dk)
    if debug: print(range_dist)
    if range_dist != float('inf'):
      delta = np.random.uniform(0, range_dist)
      i.time = delta
      reduction = range_dist - delta
      difference = abs(dk-dj)
      if dj < dk: # j is shorter
        j.time = reduction
        k.time = reduction + difference
        if k.time == float('inf'):
          return EXIT_FAILURE
      else:
        k.time = reduction
        j.time = reduction + difference
        if j.time == float('inf'):
          return EXIT_FAILURE

      if debug: print(i.time)
      if i.time < 0:
          return EXIT_FAILURE
      tcpy.fix_tree()
    else:
      return EXIT_FAILURE
    Q = 1


  elif move_type==4: # Random walk
    if debug: print("rates changed step", step )
    Q = 1 # symmetric
    def alter_rates(WINDOW_SIZE, vector, normalise=False):
      for i in range(len(vector)):
        delta = np.random.uniform(-WINDOW_SIZE, WINDOW_SIZE)
        if (vector[i]+delta >= 1 or vector[i]+delta <= 0): # prevent it from getting stuck
          return EXIT_FAILURE
        vector[i]+=delta
      if normalise: vector = [i/sum(vector) for i in vector] # has to sum to 1
      if debug: print("vector is", vector)
      return vector

    randnum = np.random.choice(4)
    v = np.array([D0_cpy[0,1], D0_cpy[1,0]]).astype(object)

    if randnum == 0:
      alpha_cpy = alter_rates(WINDOW_ALPHA, alpha_cpy, True)
      if type(alpha_cpy) == type(EXIT_FAILURE) and alpha_cpy == EXIT_FAILURE:
        return EXIT_FAILURE

    elif randnum == 1:
      d_cpy = alter_rates(WINDOW_MU, d_cpy, False)
      if type(d_cpy) == type(EXIT_FAILURE) and d_cpy == EXIT_FAILURE:
        return EXIT_FAILURE

    elif randnum == 2:
      b = B_cpy.reshape(-1)
      b = alter_rates(WINDOW_B, b)
      if type(b) == type(EXIT_FAILURE)and b == EXIT_FAILURE:
        return EXIT_FAILURE
      B_cpy = b.reshape(2,4).astype(object)

    elif randnum == 3:

      v = alter_rates(WINDOW_D0, v)
      if type(v) == type(EXIT_FAILURE) and v == EXIT_FAILURE:
        return EXIT_FAILURE
    D1_cpy = B_cpy@np.transpose(np.kron(np.ones(len(B)), np.identity(len(B)))).astype(object)
    D0_cpy = np.array([0, v[0], v[1], 0]).reshape(2,2).astype(object)
    D0_cpy = D0_cpy - np.diag(D0_cpy@np.ones(len(B)) + D1_cpy@np.ones(len(B))+d_cpy).astype(object)
    temp = D0_cpy@np.ones(len(B)) + D1_cpy@np.ones(len(B))+d_cpy
    np.testing.assert_allclose(np.array(temp).astype(np.float64), 0, atol=1e-7)

  if tcpy.head.is_neg():
    return EXIT_FAILURE
  return tcpy, Q, alpha_cpy, d_cpy, D0_cpy, B_cpy
  
from decimal import *
should_break = 0
def run_chain(s, N, t, Q1, alpha, d, D0, B, Pi, by='io', fname=None, pos = 1, send_tree=False, multip = True):
  log = 1

  chain1a = []
  chain1b = []
  chain1c = []
  if send_tree:
    t_cur = s
  else:
    t_cur = Tree(1)
    t_cur.str2tree(s,t,by=by)
  t_cur.disp(log, fname)
  p1 = tree_posterior(t_cur, alpha, d, D0, B, Q1, Pi, debug=False, fname = fname, multip=multip)
  print(p1, file=fname)


  chain1a.append(t_cur.toStr())
  chain1b.append(p1)
  chain1c.append((dcpy(alpha), dcpy(d), dcpy(D0), dcpy(B)))

  successes = 0
  for i in tqdm(range(N), position = pos):
    move = propose_move(t_cur, alpha, d, D0, B, i)
    p1 = chain1b[-1]
    if move != EXIT_FAILURE:

      t_new, q_ratio, alpha_new, d_new, D0_new, B_new = move
      # p1 = tree_posterior(t_cur, alpha, d, D0, B, Q1) # i dont need this right?
      if debug: print("operating on: ")
      if debug: t_new.disp()
      p_new = tree_posterior(t_new, alpha_new, d_new, D0_new, B_new, Q1, Pi, log, fname, multip=multip)
      
      if p_new == 1:
        print("prev tree was: ", file=fname)
        t_cur.disp(log, fname=fname)
        print(alpha, d, D0, B, file = fname)
        if should_break:
          break
        else:
          t_new = t_cur
          p_new = p1
          alpha_new = alpha
          d_new = d
          D0_new = D0
          B_new = B
      #print("q, p1, p2, ratio: ", q_ratio, p1, p_new, np.exp(p1 - p_new), file=fname)
      internal = Decimal(np.log(q_ratio) +p_new - p1)
      acc_ratio = min(np.exp(internal) , 1)
      #print("acc_ratio is: ", acc_ratio, file=fname)
      if np.random.uniform(0,1) < acc_ratio: # accept the move
        successes += 1
        if debug: print("acc at ", i, acc_ratio)
        t_cur = t_new
        p1 = p_new
        alpha = alpha_new
        d = d_new
        D0 = D0_new
        B = B_new
    chain1a.append(t_cur.toStr())
    chain1b.append(p1)
    chain1c.append((dcpy(alpha), dcpy(d), dcpy(D0), dcpy(B)))


  return successes, chain1a, chain1b, chain1c