import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


rmax = 10000
dr = 0.001
N = 1
r = np.arange(int(rmax / dr)) * dr
t = r.copy()
f = (t**2 - 1) * t**(N - 1) * np.exp(-t**2 / 2)

integral = np.zeros_like(r)

for i in range(1, len(f)):
    integral[i] = integral[i - 1] + f[i] * dr

c = -10
integral = np.log(integral + c) + (1 - N) * np.log(r) + r**2 / 2

plt.plot(r, integral)
plt.show()
