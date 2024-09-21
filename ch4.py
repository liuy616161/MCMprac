"""

import numpy
from numpy import reshape,hstack,mean,median,ptp,var,std,cov,corrcoef
import pandas as pd
df=pd.read_excel("Pdata4_6_1.xlsx",header=None)
import numpy as np
import scipy.stats as ss
from scipy import stats
a=np.array([506,508,499,503,504,510,497,512,
            514,505,493,496,506,502,509,496])
alpha=0.95; df=len(a)-1
ci=ss.t.interval(alpha,df,loc=a.mean(),scale=ss.sem(a))
print("置信区间为：",ci)
import matplotlib.pyplot as plt
import numpy as np
x=np.array([2.5, 3.9, 2.9, 2.4, 2.9, 0.8, 9.1, 0.8, 0.7,7.9, 1.8, 1.9, 0.8, 6.5, 1.6, 5.8, 1.3, 1.2, 2.7])
y=np.array([211, 167, 131, 191, 220, 297, 71, 211, 300, 107,
   167, 266, 277, 86, 207, 115, 285, 199, 172])
xy=np.array([x,y])
print(xy)
xy=xy.T[np.lexsort(xy[::-1,:])].T#array按第一行顺序排列
print(xy)
x=xy[0]
y=xy[1]
print(x,y)
plt.plot(x,y,'+k',label='原始数据点')
plt.legend()
p1=np.polyfit(x,y,deg=1)
p2=np.poly1d(np.polyfit(x,y,deg=2)); p2=p2(x)
p3=np.poly1d(np.polyfit(x,y,deg=3))(x)
p4=np.poly1d(np.polyfit(x,y,deg=4))(x)
p5=np.poly1d(np.polyfit(x,y,deg=5))(x)
plt.rc('font',size=16); plt.rc('font',family='SimHei')
plt.plot(x,np.polyval(p1,x),'b',label='一阶')
plt.plot(x,p2,'g',label='二阶')
plt.plot(x,p3,'r',label='三阶')
plt.plot(x,p4,'k',label='四阶')
#plt.plot(x,p5,'k',label='五阶')
plt.legend()
plt.savefig("figur.png",dpi=500);plt.show()
"""

"""数据处理

import pandas as pd
a=pd.read_excel("Pdata4_29.xlsx")
b1=a.fillna(0)
b2=a.fillna(method='ffill')
b3=a.fillna(method='bfill')
b4=a.fillna(value={'gender':a.gender.mode()[0],
                   'age':a.age.mean(),
                   'income':a.income.median()})
b5=a.fillna(value={'gender':a.gender.mode()[0],
                   'age': a.age.interpolate(method='polynomial',order=2),
                   'income': a.income.median()
                   })
print(b1,"\n\n",b2,"\n\n",b3,"\n\n",b4,"\n\n",b5)




"""


from scipy.optimize import linprog
c=[-1,2,3]
A=[[-2,1,1],[3,-1,-2]]; b=[[9],[-4]]
Aeq=[[4,-2,-1]]; beq=[-6]
bounds=[(-10,None),(0,None),(None,None)]
res=linprog(c,A,b,Aeq,beq,bounds)
print("目标函数的最小值：",res.fun)
print("最优解为：",res.x)



