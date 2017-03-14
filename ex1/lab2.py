import scipy.integrate as integrate
from math import cos,pi,sin
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

def cos_cornell(u):
	return integrate.quad(lambda x: cos(pi*x*x)/2, 0 ,u)

def sin_cornell(u):
	return integrate.quad(lambda x: sin(pi*x*x)/2, 0, u)

veccos = np.vectorize(cos_cornell)
vecsin = np.vectorize(sin_cornell)


us = np.random.uniform(-pi,pi, 1000)

veccos(us)
vecsin(us)

plt.scatter(veccos(us), vecsin(us), s = 1)
plt.show()
