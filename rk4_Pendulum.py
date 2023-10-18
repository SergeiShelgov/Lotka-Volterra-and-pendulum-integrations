import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def pendrhs( z):
    q = z[0]
    p = z[1]
    return np.array([ p, -np.sin(q)])

def simple_pendulum(theta_thetadot, t):
    theta, theta_dot = theta_thetadot
    return [theta_dot, - np.sin(theta)]

def stepRK4( xk, h) :
    k1 = h * pendrhs( xk)
    k2 = h * pendrhs( xk + k1 / 2.)
    k3 = h * pendrhs( xk + k2 / 2.)
    k4 = h * pendrhs( xk + k3)
    return xk + 1./6. * (k1 + 2.*k2 + 2.*k3 + k4)

t = np.linspace(0, 5 * np.pi, 1000)

theta_init = np.linspace(0, np.pi, 20)

plt.figure(1)

for theta_0 in theta_init:
    theta, theta_dot = odeint(simple_pendulum, (theta_0, 0), t).T
    plt.plot(theta, theta_dot,'-k',linewidth=0.6)

def simulate( infiniStep, inival, interval=[0, 1], Nts=100) :
    t0, te = interval[0], interval[1]
    h = 1./Nts*(te-t0)
    N = inival.size

    sollist = [inival.reshape((1, N))]
    tlist = [t0]

    xk = inival

    for k in range(1, Nts+1):

        xk = infiniStep( xk, h)

        sollist.append( xk.reshape((1, N)))
        tlist.append( t0 + k*h)

    sollist = np.vstack( sollist)

    return sollist, tlist

inival = np.array([ 0., 1.1])
interval = [0, 50]##############
nSteps = 1000

sols = []

sol, tlist = simulate( stepRK4, inival, interval, nSteps)
sols.append( sol)

plt.figure(1)
for sol in sols :
    plt.plot(sol[:,0], sol[:,1],linestyle='-', marker='o', color='k',markerfacecolor='r',linewidth=0.6,ms=5)
plt.xlabel('q'), plt.ylabel('p')
plt.title( 'pendulum rk4')

plt.show()