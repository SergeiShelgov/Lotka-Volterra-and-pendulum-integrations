import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

alpha = 2
beta = 1.1
delta = 1
gamma = 1.1
x0 = 4
y0 = 2
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
    plt.title("Lotka-Volterra model symplectic Euler");

u0=0.311
v0=1.88
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.334
v0=2.2
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.33
v0=2.10
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.32
v0=1.99
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)


u0=0.302
v0=1.77
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.283
v0=1.66
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.241
v0=1.59
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.190
v0=1.59
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.148
v0=1.65
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.126
v0=1.76
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.108
v0=1.88
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.096
v0=2.0
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.087
v0=2.11
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.082
v0=2.2
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.120
v0=2.135
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.148
v0=2.10
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.171
v0=2.13
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2
v0=2.16
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.227
v0=2.16
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.255
v0=2.15
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.274
v0=2.11
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.302
v0=2.145
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0
for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.325
v0=2.04
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.316
v0=1.93
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.306
v0=1.82
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.292
v0=1.71
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.264
v0=1.62
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.213
v0=1.58
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.166
v0=1.61
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.134
v0=1.71
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.115
v0=1.82
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1
v0=1.94
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.090
v0=2.06
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.0835
v0=2.16
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.101
v0=2.156
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.332
v0=2.16
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.316
v0=2.17
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.264
v0=2.13
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.288
v0=2.13
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.241
v0=2.16
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.213
v0=2.163
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.180
v0=2.15
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.157
v0=2.115
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.129
v0=2.115
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.092
v0=2.175
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.110
v0=2.14
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2979
v0=1.738
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2740
v0=1.637
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2526
v0=1.601
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2278
v0=1.582
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2880
v0=1.685
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1204
v0=1.788
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1404
v0=1.676
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1586
v0=1.627
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1789
v0=1.597
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2017
v0=1.583
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1113
v0=1.850
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1040
v0=1.908
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1300
v0=1.734
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.0978
v0=1.969
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.0930
v0=2.028
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.0881
v0=2.083
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.0851
v0=2.134
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1401
v0=2.094
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1907
v0=2.160
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.0825
v0=2.179
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.3089
v0=1.850
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.3140
v0=1.904
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.3184
v0=1.960
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.3229
v0=2.015
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.3279
v0=2.070
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.3313
v0=2.130
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.3042
v0=1.794
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.0863
v0=2.190
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1767
v0=1.938
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1565
v0=1.880
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1372
v0=1.938
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1565
v0=1.994
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2419
v0=1.938
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2625
v0=1.880
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2827
v0=1.938
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2625
v0=1.994
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1722
v0=1.967
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1669
v0=1.986
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1620
v0=1.994
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1515
v0=1.994
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1451
v0=1.984
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1405
v0=1.964
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1408
v0=1.915
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1521
v0=1.885
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1466
v0=1.896
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1628
v0=1.888
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1681
v0=1.897
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.1728
v0=1.911
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2457
v0=1.961
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2492
v0=1.978
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2557
v0=1.990
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2691
v0=1.990
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2745
v0=1.981
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2797
v0=1.963
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2799
v0=1.915
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2745
v0=1.899
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2688
v0=1.887
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2566
v0=1.887
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2514
v0=1.895
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

u0=0.2464
v0=1.914
h = 0.12
N=50
u = np.empty(N)
v = np.empty(N)
u[0] = u0
v[0] = v0

for n in range(N-1):
    v[n + 1] = v[n] / (1 + h * (u[n] - 1))
    u[n + 1] = u[n] + h * u[n] * (v[n + 1] - 2)
plt.figure(1)
mask = N = [0,5,10,14,18]
plt.plot(u[mask],v[mask], 'ro',mec='k',mew=1,ms=5)

plt.show()