import numpy as np 
import scipy.integrate
from math import sin,pi,cos, sqrt

import numpy as np
import matplotlib.pyplot as plt

m = 1
g = 9.81
l = g
q = 0
F = 0


def derivatives(y,t, omega_d):
	return [y[1], -sin(y[0])-q*y[1]+F*sin(omega_d*t)]

num_oscillations = 10
t = np.linspace(0.0, num_oscillations*2*pi, sqrt(num_oscillations)*num_oscillations*2*pi)

y0 = [0.01, 0.0]
omega_d = 2/3
y = scipy.integrate.odeint(derivatives, y0, t, args=(omega_d,))

thetas = np.array(y)

energy = []
for i in range(len(y)):
	theta = y[i,0]
	omega = y[i,1]
	energy.append(m*l*l*(1-cos(theta) + omega*omega/2.0))


fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=False)

ax0.plot(t, thetas[:,0], '-')
ax0.set_title(str(num_oscillations) + " oscillation + theta_0:" + str(round(y0[0],2)) + " omega_0: " +str(y0[1]))

ax1.plot(t, energy, '-')
ax1.set_title('Energy vs time')

plt.show()