import numpy as np
import pandas as pd
from matplotlib.pyplot import *
x=np.linspace(0,10*np.pi,1000)
rc('font',size=16) #rc('text',usetex=True)
for i in range(1,7):
    y=i*x*x*np.sin(x)+2*i+np.cos(x**3)
    ax=subplot(2,3,i)
    ax.plot(x,y,'b',label='$%dx^2sin(x)+%d+cos(x^3)$'%(i,2*i))
    legend()
savefig('figure.png',dpi=500); show()