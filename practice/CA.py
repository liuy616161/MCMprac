import numpy as np
import sympy as sp
import random
import seaborn as sbn
import matplotlib.pyplot as plt
model={}
size={}
def init_model(shape):
    if shape=="ridges":

    if shape=="streamline":

    if shape=="oval":


def Surfsand(current_model,watermode,left_probability,diff_dist,size):
    model=current_model
    #模拟水源的出现
    #watermode=1为朝岸，0为浸泡，-1为离岸
    #model=0为空，1为沙子，2为水
    if watermode==1:
        for i in range(size/10):
            model[0][random.random()*size][random.random()*size]=2
    if watermode==-1:
        for i in range(size / 10):
            model[size-1][random.random() * size][random.random()*size] = 2
    if watermode==0:
        for i in range(size/80):
            model[0][random.random()*size][random.random()*size]=2
            model[size - 1][random.random() * size][random.random()*size] = 2
            model[random.random() * size][0][random.random() * size] = 2
            model[random.random() * size][size-1][random.random() * size] = 2
    #模拟水流效果
    #水源对周围进行流动
    for i in range(size):
        for j in range(size):
            for k in range(size):
                waterflow(i,j,k,size)
    #模拟水源对沙子效果
    #left_probability为最近时离开的几率
    #diff_dist为水源的作用范围
    for i in range(size):
        for j in range(size):
            for k in range(size):
                if model[i][j][k]==2:
                    for d in range(diff_dist):
                        d=d+1
                        for x in (-1,0,1):
                            for y in (-1, 0, 1):
                                for z in (-1, 0, 1):
                                    if (0 <= (i + d*x) < size
                                            and 0 <= (j + d*y) < size
                                            and 0 <= (k + d*z) < size):
                                        if model[i+d*x][j+d*y][k+d*z]==1:
                                            if random.random()>left_probability/(sp.sqrt(x*x+y*y+z*z)*d**3):
                                                model[i+d*x][j+d*y][k+d*z]=0
    #对底层情况砂层进行判断，小于固定比例则终结
    over=0
    for k in range(size):
        cnt=0
        for i in range(size):
            for j in range(size):
                if model[i][j][k]==1:
                    cnt=cnt+1
        if cnt/(size*size)<k/size*0.6:
            over=1
        if over==1:
            break
    if over==1:
        return
def waterflow(i,j,k,size):
    #递归实现水流流动
    if model[i][j][k] == 2:
        for x,y,z in ((-1,0,0),(1,0,0),
                      (0,1,0),(0,-1,0),
                      (0,0,1),(0,0,-1)):
            if (0 <= (i + x) < size
                    and 0 <= (j + y) < size
                    and 0 <= (k + z) < size):
                if model[i + x][j + y][k + z] == 0:
                    model[i + x][j + y][k + z] = 2
                    waterflow(i + x, j + y, k + z, size)

