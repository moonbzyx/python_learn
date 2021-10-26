import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1)
b = np.cos(x)
y = np.sin(x)

plt.ion()  # interactive mode on
ax = plt.gca()
ax.plot(x, b, 'r')  # the first curve
ax.plot(x, 2*b, 'b')  # thes second curve
line, = ax.plot(x, y)  # the third curve and its return values
ax.set_ylim([-5, 5])

# line.set_ydata(3*y)
# plt.draw()
# plt.show()

for i in np.arange(100):
    line.set_ydata(y)
    plt.draw()
    y = y*1.01
    plt.pause(0.1)
