import numpy as np
from espirit import calreg, calmat

def estvar(X, k, r):

  C = calreg(X, r)
  A = calmat(C, k, r)

  sx = C.shape[0]
  sy = C.shape[1]
  sz = C.shape[2]

  p = A.shape[0]
  q = A.shape[1]

  d = (sx > 1) + (sy > 1) + (sz > 1)

  U, S, VH = np.linalg.svd(A, full_matrices=True)

  #return (np.sum(np.power(S[-r:], 2)))/r
  return np.sum(S)/len(S)
