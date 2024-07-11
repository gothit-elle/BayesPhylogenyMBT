import numpy as np
def build_mtrx(mu0, mu1, q01, q10, lambda0, lambda1):
  # this function builds our parameter matrices
  d = [mu0, mu1]
  # q00 = -(mu0+q01+lambda0)
  # q11 = -(mu1+q10+lambda1) # this is because D0@np.ones(2) + D1@np.ones(2)+d = 0
  D0 = np.array([0, q01, q10, 0]).reshape(2,2).astype(object)
  B = np.array([(lambda0, 0,0,0,0,0,0,lambda1)]).reshape(2,4).astype(object)
  D1 = B@np.transpose(np.kron(np.ones(len(B)), np.identity(len(B)))).astype(object) # is this sus?
  D0 = D0 - np.diag(D0@np.ones(len(B)) + D1@np.ones(len(B))+d).astype(object)
  temp = D0@np.ones(len(B)) + D1@np.ones(len(B))+d
  np.testing.assert_allclose(np.array(temp).astype(np.float64), 0, atol=1e-7)
  return [d, D0, D1, B]