"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
x=np.arange(0,25,2)
y=np.array([12,9,9,10,18,24,28,27,25,20,18,15,13])
xnew=np.linspace(0,24,500)
f1=interp1d(x,y); y1=f1(xnew)
f2=interp1d(x,y,'cubic'); y2=f2(xnew)
plt.rc('font',size=16); plt.rc('font',family='SimHei')
plt.subplot(121),plt.plot(xnew,y1); plt.xlabel("分段插值")
plt.subplot(122); plt.plot(xnew,y2); plt.ylabel("三次插值")
plt.show()

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import  numpy as np
from numpy.linalg import norm
from scipy.interpolate import interp2d
x=np.arange(0,1500,100)
y=np.arange(1200,-100,-100)
z=np.loadtxt("Pdata7_5.txt")
f=interp2d(x,y,z,'cubic')
xn=np.linspace(0,1400,141)
yn=np.linspace(0,1200,121)
zn=f(xn,yn)
m=len(xn); n=len(yn); S=0
for i in np.arange(m-1):
    for j in np.arange(n-1):
        p1=np.array([xn[i],yn[j],zn[j,i]])
        p2=np.array([xn[i+1],yn[j],zn[j,i+1]])
        p3=np.array([xn[i+1],yn[j+1],zn[j+1,i+1]])
        p4=np.array([xn[i],yn[j+1],zn[j+1,i]])
        p12=norm(p1-p2);p23=norm(p2-p3);p34=norm(p3-p4)
        p14=norm(p1-p4);p13=norm(p1-p3)
        L1=(p12+p23+p13)/2; S1=np.sqrt(L1*(L1-p12)*(L1-p23)*(L1-p13))
        L2 = (p13 + p14 + p34) / 2
        S2 = np.sqrt(L2 * (L2 - p13) * (L2 - p14) * (L2 - p34))
        S=S+S1+S2
print("面积是：",S)
plt.rc('font',size=16)
plt.subplot(121); contr=plt.contour(xn,yn,zn); plt.clabel(contr)
plt.xlabel=('$x$'); plt.ylabel('$y$',rotation=90)
ax=plt.subplot(122,projection='3d')
X,Y=np.meshgrid(xn,yn)
ax.plot_surface(X,Y,zn,cmap='viridis')
ax.set_xlabel('$x$');ax.set_ylabel('$y$');ax.set_zlabel('$z$')
plt.savefig('fefef',dpi=500); plt.show()
"""

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.interpolate import interp1d,interp2d


"""x=np.array([0,2,4,5,6,7,8,9,10.5,11.5,12.5,14,16,17,18,19,20,21,22,23,24])
y=np.array([2,2,0,2,5,8,25,12,5,10,12,7,9,28,22,10,9,11,8,9,3])
f1=interp1d(x,y,'cubic')
xnew=np.linspace(0,24,24*60)
print(xnew)
ynew=f1(xnew)
plt.plot(xnew,ynew)
plt.show()
print(ynew.sum())"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sympy import *
x=np.linspace(0,18,19)

y=np.array([9.6,18.3 ,29.0 ,47.2 ,71.1 ,119.1 ,174.6 ,257.3 ,350.7 ,441.0 ,513.3 ,559.7 ,594.8 ,629.4 ,640.8 ,651.1 ,655.9 ,659.6 ,661.8 ])
f1=interp1d(x,y,'cubic',fill_value="extrapolate")
xnew=np.linspace(0,20,20*100)
ynew=f1(xnew)
fig=plt.figure()
ax1=fig.add_subplot(1,2,1)
ax1.plot(xnew,ynew)
ynew={}
for i in range(len(xnew)):
    ynew[i]=0.06531*(665-xnew[i])*xnew[i]
result=ynew.items()
ynew=list(result)
ynew=np.array(ynew).reshape(2000,-1)
ynew=list(ynew[:,1])
print(xnew,"\n",ynew)
ax2=fig.add_subplot(1,2,2)
ax2.plot(xnew,ynew)
plt.show()


f=lambda x,k:k*(665-x)*x
popt,pcov=curve_fit(f,x,y)
print("K:",popt)
"""

from numpy import pi
x=np.array([0,3316,6635,10619,13937,
            17921,21240,25223,28543,32284])
y=np.array([3175,3110,3054,2994,2947,2892,2850,
            2795,2752,2697])
E=30.24

def ff(y):
    yy=((57/2*E)**2)*pi*y/100*E
    return yy
yy=ff(y)

print(yy)"""


"""
x=np.array([7.0,10.5,13.0,17.5,34.0,40.5,44.5,48.0,56.0,
            61.0,68.5,765,80.5,91.0,96.0,101.0,104.0,106.5,
            111.5,118.0,123.5,136.5,142.0,146.0,150.0,157.0,158.0])
y1=np.array([44,45,47,50,50,38,30,30,34,36,34,41,45,
            46,43,37,33,28,32,65,55,54,52,50,66,66,68])
y2=np.array([44,59,70,72,93,100,110,110,110,117,118,116,118,
             118,121,124,121,121,121,122,116,83,81,82,86,85,68])
f1=interp1d(x,y1,'cubic',fill_value="extrapolate")
f2=interp1d(x,y2,'cubic',fill_value="extrapolate")
xnew=np.linspace(7,158,1520)
y1new=f1(xnew)
y2new=f2(xnew)
yy=(y2new-y1new).sum()
yy=yy/18*40/18*40
print(yy)
plt.plot(xnew,y1new)
plt.plot(xnew,y2new)
plt.show()"""


"""import sympy as sp
t=sp.symbols('t')
x1,x2,x3=sp.symbols('x1,x2,x3',cls=sp.Function)
eq=[x1(t).diff(t)-2*x1(t)+3*x2(t)-3*x3(t),
    x2(t).diff(t)-4*x1(t)+5*x2(t)-3*x3(t),
    x3(t).diff(t)-4*x1(t)+4*x2(t)-2*x3(t)]
con={x1(0):1,x2(0):2,x3(0):3}
s=sp.solve(eq,ics=con)
print(s)

r,x0,xm,t,t0=sp.symbols('r,x0,xm,t,t0')
x=sp.symbols('x',cls=sp.Function)
eq2=[x(t).diff(t)-r*(1-x(t)/xm)*x(t)]
con1={x(t0):x0}
ss=sp.solve(eq2,x(t),ics=con1)
sp.pprint(ss)

from scipy.integrate import odeint
import numpy as np
from mpl_toolkits import mplot3d
import  matplotlib.pyplot as plt
def lorenz(w,t):
    sigma=10; rho=28; beta=8/3
    x,y,z=w"""



"""
import numpy as np
from scipy.optimize import curve_fit
a=[],b=[]
with open("Pdata8_10_1.txt") as f:
    s=f.read().splitlines()
for i in range(0,len(s),2):
    d1=s[i].split("\t")
    for j in range(len(d1)):
        if d1[j]!=" ": a.append(eval(d1[j]))
for i in range(1,len(s),2):
    d2=s[i].split("\t")"""

"""
from sympy import sin
import sympy as sp
x,y=sp.symbols('x,y',cls=sp.Function)
t=sp.symbols('t')
eq=sp.diff(y(t),t,1)-2*(y(t)**2)-1
con={y(0):0}
sp.pprint(sp.dsolve(eq,y(t),ics=con))

eq=sp.diff(y(t),t,3)-2*sp.diff(y(t),t,2)+y(t)
con={y(0):1,sp.diff(y(t),t).subs(t,0):1,sp.diff(y(t),t,2).subs(t,0):0}
sp.pprint(sp.dsolve(eq,y(t),ics=con))

eq=[sp.diff(x(t),t,1)-x(t)+2*y(t),
    sp.diff(y(t),t,1)-x(t)-2*y(t)]
con={x(0):1,y(0):0}
sp.pprint(sp.dsolve(eq,ics=con))"""

"""
t=Symbol('t')
#xt=Symbol('xt',cls=Function)
xt=-(sqrt(7)*E*E*sin(sqrt(7)*t/2)/7+exp(3*t/2))
plot(t,xt)"""

import matplotlib.pyplot as plt

