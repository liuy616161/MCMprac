import numpy as np
import pandas as pd
from matplotlib.pyplot import *
fig=figure()
ax=axes(projection='3d')
t=np.linspace(0,2*np.pi,50)
s=np.linspace(-1,1,10)
print("BEFORE\tt:",t,"\ts:",s)
t,s=np.meshgrid(t,s)
print("ON\tt:",t,"\ts:",s)
t=t.flatten()
s=s.flatten()
print("END\tt:",t,"\ts:",s)
x=(2+(s/2)*np.cos(t/2))*np.cos(t)
y=(2+(s/2)*np.cos(t/2))*np.sin(t)
z=(s/2)*np.sin(t/2)*0.5

import matplotlib.tri as mtri
tri = mtri.Triangulation(s, t)

ax.plot_trisurf(x,y,z,cmap="cool",triangles=tri.triangles)
ax.set_zlim(-1, 1)
show()