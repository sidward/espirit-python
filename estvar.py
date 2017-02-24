import numpy as np
import matplotlib.pyplot as plt
from espirit import calreg, calmat, ifft

def estvar(X, k, r, method):
  if (method == 1):
    return estvar_cov(X, k, r)
  elif (method == 2):
    return estvar_meansub(X, k, r)
  elif (method == 3):
    return estvar_patches(X, k, r)
  return -1

def estvar_cov(X, k, r):
  C = calreg(X, r)
  c = ifft(C, (0, 1, 2))
  
  vec = np.reshape(c, (c.shape[0] * c.shape[1] * c.shape[2], c.shape[3]))
  u = np.mean(vec, 1)
  u.shape = (len(u), 1)
  vec = vec - np.tile(u, (1, c.shape[3]))
  cov = vec.conj().T.dot(vec)
  d, v = np.linalg.eig(cov)
  d = (np.abs(d)/vec.shape[0])
  d.sort()
  assert (d[0] < d[-1])
  d = d[:int(np.floor(c.shape[3] * 0.4))]
  d = d[d > 1e-6]
  return np.mean(d)

def estvar_meansub(X, k, r):
  C = calreg(X, r)
  c = ifft(C, (0, 1, 2))

  vec = np.reshape(c, (c.shape[0] * c.shape[1] * c.shape[2], c.shape[3]))
  u = np.mean(vec, 1)
  u.shape = (len(u), 1)
  vec = vec - np.tile(u, (1, c.shape[3]))
  #vec = np.power(np.abs(vec.flatten()), 2)
  return np.var(vec)

def estvar_patches(X, k, r):

  # http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Chen_An_Efficient_Statistical_ICCV_2015_paper.pdf

  C = calreg(X, r)
  c = ifft(C, (0, 1, 2))

  sx = np.shape(c)[0]
  sy = np.shape(c)[1]
  sz = np.shape(c)[2]
  nc = np.shape(c)[3]

  p = (sx > 1) + (sy > 1) + (sz > 1)

  X = np.zeros((k**p * nc, (r - k + 1)**p)).astype(np.complex64)
  idx = 0
  for xdx in range(max(1, c.shape[0] - k + 1)):
    for ydx in range(max(1, c.shape[1] - k + 1)):
      for zdx in range(max(1, c.shape[2] - k + 1)):
        # numpy handles when the indices are too big
        block = C[xdx:xdx+k, ydx:ydx+k, zdx:zdx+k, :].astype(np.complex64) 
        X[:, idx] = block.flatten()
        idx = idx + 1
  
  u = np.mean(X, 1)
  #u = np.sum(X, 1)

  cov = np.zeros((X.shape[0], X.shape[0])).astype(np.complex64)
  for idx in range((r - k + 1)**p):
    vec = X[:, idx]
    dif = vec - u
    dif.shape = (len(dif), 1)
    cov = cov + dif.dot(dif.conj().T)
  cov = cov/((r-k+1)**p)

  d, v = np.linalg.eig(cov)
  d = np.abs(np.sort(d)[::-1])
  assert(d[0] > d[-1])
  d = d[:k**p]

  tau_arr = np.zeros((k**p, 1)).astype(np.complex64)
  med_arr = np.zeros((k**p, 1)).astype(np.complex64)
  for idx in range(k**p):
    tau = np.sum(d[idx:])/(k**p - idx)
    tau_arr[idx] = tau
    med_arr[idx] = np.median(d[idx:])

  diff = np.abs(med_arr - tau_arr)
  idx = np.argmin(diff)
  return np.abs(tau_arr[idx])
