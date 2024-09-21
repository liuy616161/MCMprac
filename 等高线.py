import numpy as np
import pandas as pd
from matplotlib.pyplot import *
fig=figure()
ax=axes(projection='3d')
x=np.linspace(-2,2,20)
y=np.linspace(-2,3,20)
x,y=np.meshgrid(x,y)
x=x.flatten()
y=y.flatten()
z=x*pow(np.e,-(x**2+y**2))
ax.plot_trisurf(x,y,z,cmap='cool')
show()