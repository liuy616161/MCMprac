
"""
############## sympy画图############

from sympy import *
x1,x2,x3,x4=symbols('m1:5')
from sympy.plotting import plot
from sympy.abc import x,pi
from sympy.functions import sin,cos
plot((2*sin(x),(x,-6,6)),(cos(x+pi/4),(x,-5,5)))
from pylab import rc
from sympy.plotting import plot3d
from sympy.abc import x,y
from sympy.functions import sin,sqrt
rc('font',size=16)
plot3d(sin(sqrt(x**2+y**2)),(x,-10,10),(y,-10,10),xlabel='x',
       ylabel='y')
from pylab import rc
from sympy import plot_implicit as pt,Eq
from sympy.abc import x,y
pt(Eq((x-1)**2+(y-2)**3,4),(x,-6,6),(y,-2,4),xlabel='x',ylabel='y')"""
import numpy as np

"""
  ######## 微积分解析解##########
from sympy import *
x,y=symbols('x y')
z=sin(x)+x**2*exp(y)
print("2阶偏导",diff(z,x,2))
print("1阶偏导",diff(z,y))

from pylab import rc
from sympy.plotting import *
rc('font',size=16)
x=symbols('x'); y=sin(x)
for k in range(3,8,2): print(y.series(x,0,k))

from sympy import integrate,symbols,sin,pi,oo
x=symbols('x')
print(integrate(sin(2*x),(x,0,pi)))
print(integrate((sin(x)/x),(x,0,oo)))

x,y=symbols('x y')
print(roots((x-2)**2*(x-1)**3,x))

"""
"""
线代的形式解

import sympy as sp
A=sp.Matrix([[1,-5,2,-3],[5,3,6,-1],[2,4,2,1]])
print("A的基础解系为",A.nullspace())

A=sp.Matrix([[1,1,-3,-1],[3,-1,-3,4],[1,5,-9,-8]])
b=sp.Matrix([1,4,0]); b.transpose()
C=A.row_join(b)
print("增广阵的行最简形:",C.rref())

A=sp.Matrix([[0,-2,2],[-2,-3,4],[2,4,-3]])
print("A的特征值为",A.eigenvals())
print("A的特征向量为",A.eigenvects())
"""
"""练习

from sympy import *
x=symbols('x')
y=symbols('y',cls=Function)
print(integrate(sqrt(1+4*x),(x,0,1)))
print(integrate(exp(-x)*sin(x),(x,0,oo)))

f=x**3-4*x**2+6*x-8
print(len(solve(f,x)))
print(solve(f,x)[0].evalf())

eq1=diff(y(x),x,2)+y(x)-x*cos(2*x)
print(dsolve(eq1,y(x),ics={y(0):1,y(2):3}))

xy=np.array([[0,0],[20,0]])
import matplotlib.pyplot as plt
import numpy.linalg as ng
v=np.array([4.5,3])
time=8.2; divs=201
T=np.linspace(0,time,divs); dt=T[1]-T[0]
Txy=xy; xyn=np.empty((2,2))
for n in range(1,len(T)):
    dxy=xy[1]-xy[0]; dd=dxy/ng.norm(dxy)
    xyn[0]=xy[0]+v[0]*dt*dd
    xyn[1]=([xy[1][0],xy[1][1]+v[1]*dt])
    if(n<=2):
        print("xyn0:\r",xyn[0])
        print("xyn1:\r",xyn[1])
    Txy=np.c_[Txy,xyn]; xy=xyn.copy()

for i in range(2):
    plt.plot(Txy[i,::2],Txy[i,1::2])
plt.savefig("fin",dpi=800); plt.show()



"""

from sympy.plotting import plot,plot_implicit
from sympy.abc import x,y
from sympy import solve
#plot((10*(0.6*((1-x/20)**(5/3))-3*((1-x/20)**(1/3))))+24,(x,0,20))
plot_implicit(x**2+y**2-4*y,(x,-5,5),(y,1,3))

