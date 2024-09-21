import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(0,100,101)

y=50/(x+10)+10
plt.plot(y,x)
plt.show()
