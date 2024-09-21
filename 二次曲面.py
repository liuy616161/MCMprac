import numpy as np
import pandas as pd
from matplotlib.pyplot import *
x=np.linspace(-10,10,100)
y=np.linspace(-10,10,100)
X,Y=np.meshgrid(x,y)
Z=np.sqrt(0.75*X*X+0.6*Y*Y-6)
ax1=subplot(1,1,1,projection='3d')
ax1.plot_surface(X,Y,Z,cmap='viridis')
ax1.set_xlabel('x');ax1.set_ylabel('y');ax1.set_zlabel('z')
savefig('figure11',dpi=500)
show()
