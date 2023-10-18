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

X0=[0.311,1.88]
t = np.linspace(0, 30, 1000)###############
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1,0], X[0:1,1], 'ro',mec='k',mew=1,ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro',mec='k',mew=1,ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro',mec='k',mew=1,ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro',mec='k',mew=1,ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro',mec='k',mew=1,ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro',mec='k',mew=1,ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.334,2.2]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.33,2.1]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.32,1.99]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.302,1.77]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.283,1.66]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.241,1.59]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.190,1.59]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.148,1.65]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = len(t)
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.126,1.76]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.108,1.88]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.096,2.0]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.087,2.11]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.082,2.2]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.120,2.135]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.148,2.10]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.171,2.13]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2,2.16]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.227,2.16]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.255,2.15]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.274,2.11]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.302,2.145]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.325,2.04]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.316,1.93]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.306,1.82]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.292,1.71]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.264,1.62]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.213,1.58]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.166,1.61]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.134,1.71]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.115,1.82]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1,1.94]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.090,2.06]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.0835,2.16]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.101,2.156]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.332,2.16]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.316,2.17]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.264,2.13]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.288,2.13]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.241,2.16]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.213,2.163]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.180,2.15]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.157,2.115]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.129,2.115]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.092,2.175]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.110,2.14]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2979,1.738]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2740,1.637]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2526,1.601]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2278,1.582]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2880,1.685]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1204,1.788]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1404,1.676]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1586,1.627]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1789,1.597]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2017,1.583]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.113,1.850]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1040,1.908]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1300,1.734]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.0978,1.969]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.0930,2.028]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.0881,2.083]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.0851,2.134]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1401,2.094]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1907,2.160]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.0825,2.179]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.3089,1.850]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.3140,1.904]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.3184,1.960]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.3229,2.015]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.3279,2.070]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.3313,2.130]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.3042,1.794]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.0863,2.190]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1767,1.938]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1565,1.880]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1372,1.938]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1565,1.994]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2419,1.938]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2625,1.880]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2827,1.938]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2625,1.994]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1722,1.967]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1669,1.986]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1620,1.994]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1515,1.994]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1451,1.984]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1405,1.964]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1408,1.915]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1521,1.885]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1466,1.896]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1628,1.888]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1681,1.897]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.1728,1.911]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2457,1.961]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2492,1.978]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2557,1.990]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2691,1.990]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2745,1.981]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2797,1.963]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2799,1.915]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2745,1.899]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2688,1.887]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2566,1.887]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2514,1.895]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

X0=[0.2464,1.914]
def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = 150
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    plt.plot(X[0:1, 0], X[0:1, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[30:31, 0], X[30:31, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[55:56, 0], X[55:56, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[80:81, 0], X[80:81, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[110:111, 0], X[110:111, 1], 'ro', mec='k', mew=1, ms=5)
    plt.plot(X[135:136, 0], X[135:136, 1], 'ro', mec='k', mew=1, ms=5)
    return X
Xrk4 = RK4(derivative, X0, t, alpha, beta, delta, gamma)

plt.show()