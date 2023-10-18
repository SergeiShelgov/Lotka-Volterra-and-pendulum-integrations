import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

Nt = 1000
tmax = 30.
t = np.linspace(0.,tmax, Nt)
def derivative(X, t):
    u, v = X
    dotu = u * (v-2)
    dotv = v * (1-u)
    return np.array([dotu, dotv])

plt.figure(1)
IC = np.linspace(1.0, 6.0, 14)
for deer in IC:
    X0 = [deer, 1.0]
    Xs = integrate.odeint(derivative, X0, t)
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