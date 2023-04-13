import numpy as np
from matplotlib import pyplot as plt
import math

x = np.linspace(0.01, 100)
l = x/(x+1)

r1 = ((np.cos(x) - 1.33 * np.sqrt(1 - (1 / 1.33 * np.sin(x)) ** 2)) / (
            np.cos(x) + 1.33 * np.sqrt(1 - (1 / 1.33 * np.sin(x)) ** 2))) ** 2
r2 = (np.sqrt(1 - (1 / 1.33 * np.sin(x)) ** 2 - 1.33 * np.cos(x)) / np.sqrt(
    1 - (1 / 1.33 * np.sin(x)) ** 2 + 1.33 * np.cos(x))) ** 2

for i in range(50):
    if math.isnan(r2[i]):
        r2[i] = 0

Rr = 0.5 * (r1 + r2)

plt.figure(1)
plt.figure(figsize=(9, 6))

plt.xlabel(r"$l/m$", fontdict={'weight': 'normal', 'size': 20})
plt.ylabel(r"$\mathregular{R_r}$", fontdict={'weight': 'normal', 'size': 20})

# plt.ylim(-0.1, 1)
# plt.title("Reflectance")
# plt.annotate('max', xy=(0.22, 0.9), xytext=(0.22, 0.5),arrowprops=dict(facecolor='black'))

plt.plot(x, l)  

plt.show()