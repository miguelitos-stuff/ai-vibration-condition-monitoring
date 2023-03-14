import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,10, 50)
y = np.sin(x)

plt.figure()
plt.plot(x,y)
plt.show()
