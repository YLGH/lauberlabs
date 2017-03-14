import numpy as np 
import scipy.integrate
from math import sin,pi,cos, sqrt

import numpy as np
import matplotlib.pyplot as plt

m = 1
g = 9.81
l = g
q = 0.0
f = 0.5


def derivatives(y,t, omega_d):
	return [y[1], -sin(y[0])-q*y[1]+f*sin(omega_d*t)]

num_oscillations = 50
t = np.linspace(0.0, num_oscillations*2*pi, sqrt(num_oscillations)*num_oscillations*2*pi)

y0 = [pi/3, 0.0]
omega_d = 2/3
y = scipy.integrate.odeint(derivatives, y0, t, args=(omega_d,))

thetas = np.array(y)

energy = []
for i in range(len(y)):
	theta = y[i,0]
	omega = y[i,1]
	energy.append(m*l*l*(1-cos(theta) + omega*omega/2.0))


# fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=False)

# ax0.plot(t, thetas[:,0], '-')
# ax0.set_title(str(num_oscillations) + " oscillation theta_0:" + str(round(y0[0],5)) + " omega_0: " + str(y0[1]) + " (q,f) : " + str((q,f)))

# ax1.plot(t, thetas[:,1], '-')
# ax1.set_title('Angular velocityvs time')

# plt.savefig(str(num_oscillations) + "oscillation"+"theta_0:" + str(round(y0[0],5)) + "omega_0: "+ str(y0[1]) + " (q,f) : " + str((q,f))+'.png' )
# plt.show()

fig, (ax0) = plt.subplots(nrows=1, sharex=False)


ax0.plot(thetas[:,0],  thetas[:,1], '-')
ax0.set_title('Angle vs Angle velocity num_oscillations: ' + str(num_oscillations) + " oscillation theta_0:" + str(round(y0[0],5)) + " omega_0: " + str(y0[1]) + " (q,f) : " + str((q,f)))


plt.savefig('thetavsvelocity (q,f,theta_0): ' + str((q,f,y0[0])) +'.png' )
plt.show()

