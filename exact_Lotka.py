import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

alpha = 2
beta = 1.1
delta = 1.
gamma = 1.1
Nt = 1000
tmax = 30.
t = np.linspace(0,tmax, Nt)

def derivative(X, t, alpha, beta, delta, gamma):
    x, y = X
    dotx = x * (alpha - beta * y)
    doty = y * (-delta + gamma * x)
    return np.array([dotx, doty])

plt.figure(1)
IC = np.linspace(1.0, 6.0, 14)
for deer in IC:
    X0 = [deer, 1.0]
    Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
    plt.plot(Xs[:,0], Xs[:,1], "-k", linewidth=0.6)
    plt.xlabel("Deer")
    plt.ylabel("Wolves")
    plt.title("Lotka-Volterra exact flow");

X0 = [0.311,1.88]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.334,2.2]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.33,2.1]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.32,1.99]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.302,1.77]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.283,1.66]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.241,1.59]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.190,1.59]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.148,1.65]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.126,1.76]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.108,1.88]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.096,2.0]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.087,2.11]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.082,2.2]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.120,2.135]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.148,2.10]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.171,2.13]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.2,2.16]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.227,2.16]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.255,2.15]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.274,2.11]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.302,2.145]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.325,2.04]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.316,1.93]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.306,1.82]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.292,1.71]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.264,1.62]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.213,1.58]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.166,1.61]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.134,1.71]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.115,1.82]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.1,1.94]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.090,2.06]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.0835,2.16]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.101,2.156]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.332,2.16]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.316,2.17]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.264,2.13]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.288,2.13]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.241,2.16]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.213,2.163]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.180,2.15]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.157,2.115]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.129,2.115]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.092,2.175]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.110,2.14]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.2979,1.738]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.2740,1.637]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.2526,1.601]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.2278,1.582]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.2880,1.685]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.1204,1.788]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.1404,1.676]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.1586,1.627]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.1789,1.597]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.2017,1.583]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.113,1.850]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.1040,1.908]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.1300,1.734]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.0978,1.969]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.0930,2.028]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0 = [0.0881,2.083]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.0851,2.134]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.1401,2.094]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.1907,2.160]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.0825,2.179]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.3089,1.850]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.3140,1.904]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.3184,1.960]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.3229,2.015]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.3279,2.070]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.3313,2.130]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.3042,1.794]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.0863,2.190]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.1767,1.938]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.1565,1.880]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.1372,1.938]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.1565,1.994]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.2419,1.938]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.2625,1.880]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.2827,1.938]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.2625,1.994]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.1722,1.967]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.1669,1.986]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.1620,1.994]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.1515,1.994]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.1451,1.984]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.1405,1.964]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.1408,1.915]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.1521,1.885]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.1466,1.896]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.1628,1.888]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.1681,1.897]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.1728,1.911]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.2457,1.961]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.2492,1.978]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.2557,1.990]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.2691,1.990]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.2745,1.981]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.2797,1.963]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.2799,1.915]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.2745,1.899]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.2688,1.887]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.2566,1.887]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.2514,1.895]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

X0=[0.2464,1.914]
Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
plt.plot(Xs[0:1, 0], Xs[0:1, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[30:31, 0], Xs[30:31, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[55:56, 0], Xs[55:56, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[80:81, 0], Xs[80:81, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[110:111, 0], Xs[110:111, 1], 'ro', mec='k', mew=1, ms=5)
plt.plot(Xs[135:136, 0], Xs[135:136, 1], 'ro', mec='k', mew=1, ms=5)

plt.show()