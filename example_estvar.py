import cfl
from estvar import estvar

import matplotlib.pyplot as plt
import numpy as np

# Pure noise test
size = (1, 50, 50, 8)
sigma = 10
X = np.random.normal(loc=0,scale=sigma/np.sqrt(2), size=size) + 1j * np.random.normal(loc=0,scale=sigma/np.sqrt(2), size=size) 

est = estvar(X, 6, 24)

print("True noise variance: %f" % sigma**2)
print("Estimated noise variance: %f" % est)
