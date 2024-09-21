import numpy as np
import sympy as sp
import random
import seaborn as sbn
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(100000)
size = 100
model=np.zeros((size,size))
def init_model(shape):
    #if shape == "ridges":
    #if shape == "streamline":
    if shape == "oval":
        for i in range(size):
            for j in range(size):
                if (i-size/2)**2/((size*0.4)**2)+(j-size/2)**2/((size*0.3)**2)<1.1:
                    model[i][j]=1
init_model("oval")
def Surfsand( watermode, left_probability, diff_dist):
    # 模拟水源的出现
    # watermode=1为朝岸，0为浸泡，-1为离岸
    # model=0为空，1为沙子，0.5为水
    if watermode == 1:
        for i in range(size // 10):
            model[0][int(random.random() * size)] = 0.5
    if watermode == -1:
        for i in range(size // 10):
            model[size - 1][int(random.random() * size)] = 0.5
    if watermode == 0:
        for i in range(size // 80):
            model[0][int(random.random() * size)] = 0.5
            model[size - 1][int(random.random() * size)] = 0.5
            model[int(random.random() * size)][0] = 0.5
            model[int(random.random() * size)][size - 1] = 0.5
    # 模拟水流效果
    # 水源对周围进行流动
    for i in range(size):
        for j in range(size):
                waterflow(i, j,watermode)
    # 模拟水源对沙子效果
    # left_probability为最近时离开的几率
    # diff_dist为水源的作用范围
    for i in range(size):
        for j in range(size):
                if model[i][j] == 0.5:
                    for d in range(diff_dist):
                        d = d + 1
                        for x in (-1, 0, 1):
                            for y in (-1, 0, 1):
                                    if (0 <= (i + d * x) < size
                                            and 0 <= (j + d * y) < size):
                                        if model[i + d * x][j + d * y] == 1:
                                            if random.random() < left_probability / (
                                                    sp.sqrt(x * x + y * y ) * d ** 3):
                                                model[i + d * x][j + d * y] = 0
    return model


def waterflow(i, j,watermode):
    # 递归实现水流流动
    if watermode==1:
        if i<int(size*0.4):
            return
    if watermode==-1:
        if i>int(size*0.6):
            return
    if model[i][j] == 0.5:
        for x, y in ((-1, 0), (1, 0),
                        (0, 1), (0, -1)):
            if (0 <= (i + x) < size
                    and 0 <= (j + y) < size):
                if model[i + x][j + y] == 0:
                    model[i + x][j + y] = 0.5
                    waterflow(i + x, j + y,watermode)

def relife():
    for i in range(size):
        for j in range(size):
            if model[i][j]==0.5:
                model[i][j]=0

N=5
for time in range (N):
    cnt=0
    model=Surfsand(watermode=1,left_probability=0.2,diff_dist=2)
    plt.figure(time*10+cnt,figsize=(8,8))
    sbn.heatmap(model,square=True)
    plt.savefig('F:/ameri/{}.jpg'.format(time*10+cnt))
    cnt=cnt+1
    model = Surfsand(watermode=0, left_probability=0.1, diff_dist=2)
    plt.figure(time*10+cnt,figsize=(8,8))
    sbn.heatmap(model,square=True)
    plt.savefig('F:/ameri/{}.jpg'.format(time*10+cnt))
    cnt=cnt+1
    model = Surfsand(watermode=0, left_probability=0.1, diff_dist=2)
    plt.figure(time*10+cnt,figsize=(8,8))
    sbn.heatmap(model,square=True)
    plt.savefig('F:/ameri/{}.jpg'.format(time*10+cnt))
    cnt=cnt+1
    relife()
    model = Surfsand(watermode=-1, left_probability=0.2, diff_dist=2)
    plt.figure(time*10+cnt,figsize=(8,8))
    sbn.heatmap(model,square=True)
    plt.savefig('F:/ameri/{}.jpg'.format(time*10+cnt))
    cnt=cnt+1
    relife()