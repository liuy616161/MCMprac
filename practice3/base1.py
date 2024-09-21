from matplotlib.pyplot import *
import numpy as py

def func(i):
    return i**3-1/i
x=np.linspace(0,3,200)
y1=x**3-1/x
y2=(func(1.001)-func(1.0))/0.001*(x-1)+func(1)
rc('font',size=16)
ax=subplot(1,1,1)
ax.plot(x,y1,'b',label='func')
legend()
ax.plot(x,y2,'r',label='ddd')
legend()
savefig('prac.png',dpi=500)
show()