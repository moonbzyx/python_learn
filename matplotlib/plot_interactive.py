import numpy as np
import matplotlib.pyplot as plt
# plt.ioff()
plt.ion()
for i in range(3):
    plt.plot(np.random.rand(10))
    plt.show()
    plt.pause(0.5)
