import numpy as np
import matplotlib.pyplot as plt
a1 = np.array([[1,2,3],[4,5,6]])

fig, (ax1, ax2)=plt.subplots(1,2, figsize=(10,10))
ax1.imshow(a1, cmap="gray")
print(a1.shape)
b = np.stack([a1,a1,a1], axis =-1)
print(b.shape)
ax2.imshow(b[1], cmap="gray")
print(b)
