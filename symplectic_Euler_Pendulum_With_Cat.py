import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

def simple_pendulum(theta_thetadot, t):
    theta, theta_dot = theta_thetadot
    return [theta_dot, - np.sin(theta)]

t = np.linspace(0, 5 * np.pi, 1000)

theta_init = np.linspace(0, np.pi, 20)

plt.figure(1)

for theta_0 in theta_init:
    theta, theta_dot = odeint(simple_pendulum, (theta_0, 0), t).T
    plt.plot(theta, theta_dot,'-k',linewidth=0.6)

q0 = -1.0899
p0 = 0.8855
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.figure(1)
mask = N = [0,10,22,36,49]
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)
plt.xlabel('q'), plt.ylabel('p')
plt.title( 'pendulum symplectic Euler')

q0 = -0.8671
p0 = 0.1954
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.6437
p0 = 0.8855
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -1.0881
p0 = 0.866
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -1.0847
p0 = 0.848
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -1.0530
p0 = 0.672
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -1.0077
p0 = 0.487
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9358
p0 = 0.295
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9001
p0 = 0.219
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8357
p0 = 0.219
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8038
p0 = 0.302
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.6731
p0 = 0.725
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7140
p0 = 0.546
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7657
p0 = 0.387
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7585
p0 = 0.742
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9779
p0 = 0.742
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8672
p0 = 0.773
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -1.0340
p0 = 0.806
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7071
p0 = 0.806
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9211
p0 = 0.766
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8097
p0 = 0.766
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9696
p0 = 0.384
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -1.0317
p0 = 0.581
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -1.0709
p0 = 0.765
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7382
p0 = 0.468
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.6927
p0 = 0.630
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.6556
p0 = 0.805
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -1.0609
p0 = 0.844
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -1.0048
p0 = 0.772
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7323
p0 = 0.772
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.6750
p0 = 0.846
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9495
p0 = 0.757
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8933
p0 = 0.774
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8375
p0 = 0.771
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)


q0 = -0.7835
p0 = 0.755
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9877
p0 = 0.434
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9519
p0 = 0.337
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9178
p0 = 0.253
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8188
p0 = 0.257
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7830
p0 = 0.346
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8835
p0 = 0.203
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8498
p0 = 0.203
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7511
p0 = 0.425
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7252
p0 = 0.506
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7024
p0 = 0.589
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.6822
p0 = 0.677
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.6639
p0 = 0.764
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.6490
p0 = 0.847
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.6583
p0 = 0.866
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.6901
p0 = 0.825
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7189
p0 = 0.790
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7444
p0 = 0.756
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7703
p0 = 0.749
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7961
p0 = 0.762
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8233
p0 = 0.770
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8514
p0 = 0.774
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8797
p0 = 0.774
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9065
p0 = 0.770
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9349
p0 = 0.762
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9632
p0 = 0.750
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9904
p0 = 0.756
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -1.0183
p0 = 0.788
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -1.0474
p0 = 0.824
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -1.0746
p0 = 0.866
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -1.0781
p0 = 0.807
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -1.0624
p0 = 0.716
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -1.0425
p0 = 0.625
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -1.0191
p0 = 0.535
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9744
p0 = 0.568
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9068
p0 = 0.568
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7634
p0 = 0.568
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8336
p0 = 0.568
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9397
p0 = 0.623
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9397
p0 = 0.513
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7973
p0 = 0.619
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7973
p0 = 0.513
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9677
p0 = 0.595
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9582
p0 = 0.617
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9490
p0 = 0.622
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9309
p0 = 0.622
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9222
p0 = 0.611
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9134
p0 = 0.590
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9134
p0 = 0.547
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9213
p0 = 0.531
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9306
p0 = 0.519
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9505
p0 = 0.519
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9604
p0 = 0.532
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.9665
p0 = 0.545
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8269
p0 = 0.591
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8189
p0 = 0.606
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8092
p0 = 0.615
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7874
p0 = 0.615
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7790
p0 = 0.606
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7711
p0 = 0.591
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7708
p0 = 0.548
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7791
p0 = 0.533
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.7878
p0 = 0.521
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8069
p0 = 0.518
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8170
p0 = 0.530
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

q0 = -0.8250
p0 = 0.544
h = 0.1
N=350
q = np.empty(N)
p = np.empty(N)
q[0] = q0
p[0] = p0
for n in range(N-1):
    q[n + 1] = q[n] + h * p[n]
    p[n + 1] = p[n] - h * math.sin(q[n + 1])
plt.plot(q[mask],p[mask], 'ro',mec='k',mew=1,ms=5)

plt.show()