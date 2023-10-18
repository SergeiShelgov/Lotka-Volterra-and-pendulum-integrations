import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def pendrhs( z):
    q = z[0]
    p = z[1]
    return np.array([ p, -np.sin(q)])

def stepRK4( xk, h) :
    k1 = h * pendrhs( xk)
    k2 = h * pendrhs( xk + k1 / 2.)
    k3 = h * pendrhs( xk + k2 / 2.)
    k4 = h * pendrhs( xk + k3)
    return xk + 1./6. * (k1 + 2.*k2 + 2.*k3 + k4)

def simple_pendulum(theta_thetadot, t):
    theta, theta_dot = theta_thetadot
    return [theta_dot, - np.sin(theta)]

t = np.linspace(0, 5 * np.pi, 1000)

theta_init = np.linspace(0, np.pi, 20)

plt.figure(1)

for theta_0 in theta_init:
    theta, theta_dot = odeint(simple_pendulum, (theta_0, 0), t).T
    plt.plot(theta, theta_dot,'-k',linewidth=0.6)

def simulate( infiniStep, inival, interval, Nts=100) :
    t0, te = interval[0], interval[1]
    h = 1./Nts*(te-t0)
    N = inival.size

    sollist = [inival.reshape((1, N))]

    xk = inival

    for k in range(1, Nts+1):

        xk = infiniStep( xk, h)

        sollist.append( xk.reshape((1, N)))

    sollist = np.vstack( sollist)

    return sollist
interval = [0, 50]
nSteps = 1000
integrators = [stepRK4]
integrator_labels = ['rk4']
sols = []
########################################
inival = np.array([-1.0899, 0.8855])

for integrator in integrators:
    sol = simulate(integrator, inival, interval, nSteps)
    sols.append( sol)

plt.figure(1)
for sol in sols :
    plt.plot(sol[0:1,0], sol[0:1,1],'ro',mec='k',mew=1,ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro',mec='k',mew=1,ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro',mec='k',mew=1,ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro',mec='k',mew=1,ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro',mec='k',mew=1,ms=5)
plt.xlabel('q'), plt.ylabel('p')
plt.title( 'pendulum rk4')

inival = np.array([-0.8671, 0.1954])
sols = []
for integrator in integrators:
    sol = simulate(integrator, inival, interval, nSteps)
    sols.append(sol)

plt.figure(1)
for sol in sols:
    plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

inival = np.array([-0.6437, 0.8855])
sols = []
for integrator in integrators:
    sol = simulate(integrator, inival, interval, nSteps)
    sols.append(sol)

plt.figure(1)
for sol in sols:
    plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

inival = np.array([-1.0881, 0.866])
sols = []
for integrator in integrators:
    sol = simulate(integrator, inival, interval, nSteps)
    sols.append(sol)

plt.figure(1)
for sol in sols:
    plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

inival = np.array([-1.0847, 0.848])
sols = []
for integrator in integrators:
    sol = simulate(integrator, inival, interval, nSteps)
    sols.append(sol)

plt.figure(1)
for sol in sols:
    plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

inival = np.array([-1.0530, 0.672])
sols = []
for integrator in integrators:
    sol = simulate(integrator, inival, interval, nSteps)
    sols.append(sol)

plt.figure(1)
for sol in sols:
    plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

inival = np.array([-1.0077, 0.487])
sols = []
for integrator in integrators:
    sol  = simulate(integrator, inival, interval, nSteps)
    sols.append(sol)

plt.figure(1)
for sol in sols:
    plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

inival = np.array([-0.9358, 0.295])
sols = []
for integrator in integrators:
    sol  = simulate(integrator, inival, interval, nSteps)
    sols.append(sol)

plt.figure(1)
for sol in sols:
    plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

inival = np.array([-0.9001, 0.219])
sols = []
for integrator in integrators:
    sol  = simulate(integrator, inival, interval, nSteps)
    sols.append(sol)

plt.figure(1)
for sol in sols:
    plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

inival = np.array([-0.8357, 0.219])
sols = []
for integrator in integrators:
    sol  = simulate(integrator, inival, interval, nSteps)
    sols.append(sol)

plt.figure(1)
for sol in sols:
    plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

inival = np.array([-0.8038, 0.302])
sols = []
for integrator in integrators:
    sol  = simulate(integrator, inival, interval, nSteps)
    sols.append(sol)

plt.figure(1)
for sol in sols:
    plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

inival = np.array([-0.6731, 0.725])
sols = []
for integrator in integrators:
    sol  = simulate(integrator, inival, interval, nSteps)
    sols.append(sol)

plt.figure(1)
for sol in sols:
    plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

inival = np.array([-0.7140, 0.546])
sols = []
for integrator in integrators:
    sol  = simulate(integrator, inival, interval, nSteps)
    sols.append(sol)

plt.figure(1)
for sol in sols:
    plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

inival = np.array([-0.7657, 0.387])
sols = []
for integrator in integrators:
    sol  = simulate(integrator, inival, interval, nSteps)
    sols.append(sol)

plt.figure(1)
for sol in sols:
    plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

inival = np.array([-0.7585, 0.742])
sols = []
for integrator in integrators:
    sol  = simulate(integrator, inival, interval, nSteps)
    sols.append(sol)

plt.figure(1)
for sol in sols:
    plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

inival = np.array([ -0.9779, 0.742])
sols = []
for integrator in integrators :
    sol  = simulate( integrator, inival, interval, nSteps)
    sols.append(sol)

plt.figure(1)
for sol in sols :
    plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

inival = np.array([ -0.8672, 0.773])
sols = []
for integrator in integrators :
    sol  = simulate( integrator, inival, interval, nSteps)
    sols.append(sol)

plt.figure(1)
for sol in sols :
    plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

inival = np.array([ -1.0340, 0.806])
sols = []
for integrator in integrators :
    sol  = simulate( integrator, inival, interval, nSteps)
    sols.append(sol)

plt.figure(1)
for sol in sols :
    plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

inival = np.array([-0.7071, 0.806])
sols = []
for integrator in integrators:
    sol  = simulate(integrator, inival, interval, nSteps)
    sols.append(sol)

plt.figure(1)
for sol in sols:
    plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9211, 0.766])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.8097, 0.766])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9696, 0.384])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-1.0317, 0.581])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-1.0709, 0.765])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7382, 0.468])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.6927, 0.630])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.6556, 0.805])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-1.0609, 0.844])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-1.0048, 0.772])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7323, 0.772])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.6750, 0.846])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9495, 0.757])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.8933, 0.774])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.8375, 0.771])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7835, 0.755])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9877, 0.434])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9519, 0.337])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9178, 0.253])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.8188, 0.257])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7830, 0.346])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.8835, 0.203])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.8498, 0.203])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7252, 0.506])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7024, 0.589])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.6822, 0.677])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.6639, 0.764])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.6490, 0.847])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.6583, 0.866])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.6901, 0.825])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7189, 0.790])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7444, 0.756])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7703, 0.749])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7961, 0.762])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.8233, 0.770])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.8514, 0.774])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.8797, 0.774])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9065, 0.770])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9349, 0.762])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9632, 0.750])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9904, 0.756])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-1.0183, 0.788])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-1.0474, 0.824])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-1.0746, 0.866])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-1.0781, 0.807])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-1.0624, 0.716])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-1.0425, 0.625])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-1.0191, 0.535])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9744, 0.568])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9068, 0.568])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9178, 0.253])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7634, 0.568])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.8336, 0.568])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9397, 0.623])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9397, 0.513])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7973, 0.619])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7973, 0.513])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9677, 0.595])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9582, 0.617])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9490, 0.622])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9309, 0.622])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9222, 0.611])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9134, 0.590])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9134, 0.547])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9213, 0.531])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9306, 0.519])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9505, 0.519])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9604, 0.532])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.9665, 0.545])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.8269, 0.591])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.8189, 0.606])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.8092, 0.615])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7874, 0.615])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7790, 0.606])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7711, 0.591])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7708, 0.548])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7791, 0.533])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.7878, 0.521])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.8069, 0.518])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.8170, 0.530])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

    inival = np.array([-0.8250, 0.544])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)

        inival = np.array([-0.7511, 0.425])
    sols = []
    for integrator in integrators:
        sol  = simulate(integrator, inival, interval, nSteps)
        sols.append(sol)

    plt.figure(1)
    for sol in sols:
        plt.plot(sol[0:1, 0], sol[0:1, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[19:20, 0], sol[19:20, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[43:44, 0], sol[43:44, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[73:74, 0], sol[73:74, 1], 'ro', mec='k', mew=1, ms=5)
        plt.plot(sol[98:99, 0], sol[98:99, 1], 'ro', mec='k', mew=1, ms=5)
plt.show()