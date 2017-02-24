import cfl
from estvar import estvar

import matplotlib.pyplot as plt
import numpy as np

k = 6
r = 24

var = 1000000
X = cfl.readcfl('data/knee/noisy')

#var = 100
#X = cfl.readcfl('data/brain/noisy')

print("Method 1")
est = estvar(X, k, r, 1)
print("True variance: %f" % var)
print("Estimated noise variance: %f" % est)
print("Ratio of estimated over true value: %f" % (est/var))

print("Method 2")
est = estvar(X, k, r, 2)
print("True variance: %f" % var)
print("Estimated noise variance: %f" % est)
print("Ratio of estimated over true value: %f" % (est/var))

print("Method 3")
est = estvar(X, k, r, 3)
print("True variance: %f" % var)
print("Estimated noise variance: %f" % est)
print("Ratio of estimated over true value: %f" % (est/var))
