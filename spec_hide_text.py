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

### symmetrize the array
arr = plt.imread('%s.png'%s)[:, :, 0]
m,n = arr.shape
aug = np.zeros((m+n,m+n), dtype=arr.dtype)
aug[:m,m:] = arr
aug[m:,:m] = arr.T

### enlarging the insignificant eigenvalues
vals,vecs = np.linalg.eigh(aug)
M = np.abs(vals).max()
mask = np.abs(vals) < 10

new_vals = vals.copy()
new_vals[mask] += M + 100

D = np.diag(new_vals)
Q = vecs
new_aug = Q.dot(D).dot(Q.T)

np.savetxt("hidden_text.csv", new_aug, delimiter=",")
