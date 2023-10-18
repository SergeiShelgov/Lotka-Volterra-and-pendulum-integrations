import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

def simple_pendulum(theta_thetadot, t):
    theta, theta_dot = theta_thetadot
    return [theta_dot, - np.sin(theta)]

def stepEulerExplicit( xk, h) :
    return xk + h * simple_pendulum( xk)

t = np.linspace(0, 5 * np.pi, 1000)

theta_init = np.linspace(0, np.pi, 20)

plt.figure(1)

for theta_0 in theta_init:
    theta, theta_dot = odeint(simple_pendulum, (theta_0, 0), t).T
    plt.plot(theta, theta_dot,'-k',linewidth=0.6)

q0 = 0.5
p0 = 0
h = 0.12
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n]-h * math.sin(q[n])
plt.figure(1)
plt.plot(q,p, linestyle='-', marker='o', color='k',markerfacecolor='r',linewidth=0.6,ms=5)
plt.xlabel('q'), plt.ylabel('p')
plt.title( 'pendulum explicit Euler')
plt.show()