import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

alpha = 2
beta = 1.1
delta = 1.
gamma = 1.1
Nt = 1000
tmax = 30.
t = np.linspace(0.,tmax, Nt)
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
    plt.plot(Xs[:,0], Xs[:,1], "-k",linewidth=0.6)
    plt.xlabel("Deer")
    plt.ylabel("Wolves")
    plt.title("Lotka-Volterra model explicit Euler");

u0 = 2
v0 = 2
h = 0.12
N=100
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    u[n + 1] = u[n] + h * u[n] * (v[n] - 2)
    v[n + 1] = v[n] + h * v[n] * (1 - u[n])
plt.figure(1)
plt.plot(u,v,linestyle='-', marker='o', color='k',markerfacecolor='r',linewidth=0.6,ms=5)

plt.show()