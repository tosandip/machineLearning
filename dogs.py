import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_ht = 28 + 4*np.random.randn(greyhounds)
lab_ht = 24 + 4*np.random.randn(labs)

plt.hist([grey_ht, lab_ht], stacked=True, color=['r','b'])
plt.show()
