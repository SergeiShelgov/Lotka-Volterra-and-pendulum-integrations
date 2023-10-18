import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def simple_pendulum(X0, t):
    theta, theta_dot = X0
    return [theta_dot, - np.sin(theta)]

t = np.linspace(0, 5 * np.pi, 1000)

theta_init = np.linspace(0, np.pi, 20)

plt.figure(1)
plt.xlabel('q'), plt.ylabel('p')
plt.title( 'pendulum exact flow')

for theta_0 in theta_init:
    theta, theta_dot = odeint(simple_pendulum, (theta_0, 0), t).T
    plt.plot(theta, theta_dot,'-k',linewidth=0.6)

X0 = [-1.0899, 0.8855]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.8671, 0.1954]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.6437, 0.8855]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-1.0881, 0.866]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-1.0847, 0.848]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-1.0530, 0.672]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-1.0077, 0.487]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9358, 0.295]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9001, 0.219]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.8357, 0.219]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.8038, 0.302]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.6731, 0.725]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7140, 0.546]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7657, 0.387]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7585, 0.742]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9779, 0.742]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.8672, 0.773]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-1.0340, 0.806]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7071, 0.806]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9211, 0.766]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.8097, 0.766]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9696, 0.384]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-1.0317, 0.581]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-1.0709, 0.765]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7382, 0.468]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.6927, 0.630]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.6556, 0.805]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-1.0048, 0.772]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7323, 0.772]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.6750, 0.846]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9495, 0.757]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.8933, 0.774]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.8375, 0.771]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7835, 0.755]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9877, 0.434]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9519, 0.337]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9178, 0.253]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.8188, 0.257]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7830, 0.346]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.8835, 0.203]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.8498, 0.203]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7252, 0.506]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7024, 0.589]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.6822, 0.677]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.6639, 0.764]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.6490, 0.847]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.6583, 0.866]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.6901, 0.825]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7189, 0.790]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7444, 0.756]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7703, 0.749]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7961, 0.762]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.8233, 0.770]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.8514, 0.774]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.8797, 0.774]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9065, 0.770]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9349, 0.762]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9632, 0.750]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9904, 0.756]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-1.0183, 0.788]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-1.0474, 0.824]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-1.0746, 0.866]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-1.0781, 0.807]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-1.0624, 0.716]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-1.0425, 0.625]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-1.0191, 0.535]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9744, 0.568]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9068, 0.568]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9178, 0.253]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7634, 0.568]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.8336, 0.568]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9397, 0.623]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9397, 0.513]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7973, 0.619]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7973, 0.513]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9677, 0.595]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9582, 0.617]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9490, 0.622]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9309, 0.622]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9222, 0.611]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9134, 0.590]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9134, 0.547]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9213, 0.531]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9306, 0.519]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9505, 0.519]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9604, 0.532]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.9665, 0.545]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.8269, 0.591]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.8189, 0.606]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.8092, 0.615]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7874, 0.615]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7790, 0.606]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-0.7711, 0.591]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)


X0 = [-0.7708, 0.548]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)


X0 = [-0.7791, 0.533]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)


X0 = [-0.7878, 0.521]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)


X0 = [-0.8069, 0.518]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)


X0 = [-0.8170, 0.530]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)


X0 = [-0.8250, 0.544]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)


X0 = [-0.7511, 0.425]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [-1.0605, 0.845]
Xs = odeint(simple_pendulum, X0, t)
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[61:62, 0], Xs[61:62, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[137:138, 0], Xs[137:138, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[232:233, 0], Xs[232:233, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[312:313, 0], Xs[312:313, 1], 'ro', mec='k', mew=1, ms=5)

plt.show()