import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

print('Enter the text: ', end='')
s = input()
l = len(s)

fig = plt.figure(figsize=(2*l,1))
ax = fig.add_axes((0,0,1,1))
ax.axis('off')
ax.text(0,0,s, weight='bold', size=100)
fig.savefig('%s.png'%s)
plt.close(fig)

arr = plt.imread('%s.png'%s)[:, :, 0]
m,n = arr.shape

### enlarging the insignificant eigenvalues
U,vals,Vh = np.linalg.svd(arr)
M = np.abs(vals).max()
mask = np.abs(vals) < 10

new_vals = vals.copy()
new_vals[mask] += M + 100

Sigma = np.zeros_like(arr, dtype=U.dtype)
Sigma[np.arange(m), np.arange(m)] = new_vals
new_arr = U.dot(Sigma).dot(Vh)

np.savetxt("hidden_text.csv", new_arr, delimiter=",")