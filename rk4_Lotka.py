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
    plt.title("Lotka-Volterra model rk4");

X0=[4,2]
t = np.linspace(0, 30, 1000)##############
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = len(t)
    X  = np.zeros([nt, len(X0)])
    X[0] = X0

    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return X

Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)
plt.figure(1)
plt.plot(Xrk4[:,0], Xrk4[:,1],linestyle='-', marker='o', color='k',markerfacecolor='r',linewidth=0.6,ms=5)
plt.show()