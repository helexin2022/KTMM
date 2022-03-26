from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import csv
from pandas import Series, DataFrame
import os
import math
import pandas as pd
import sympy
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import sympy

objFilePath = 'model3.obj'
with open(objFilePath) as file:
    points = [[]for i in range(9)]
    face=[[] for i in range(9)]
    ppoints=[]
    n=0
    m=0
    i=-1
    j=0
    while 1:
        line = file.readline()
        strs = line.split(" ")
        if not line:
            #print(n,m)
            break
        if strs[0]=="g":
            i=i+1
            j=j+1
            line = file.readline()
            strs = line.split(" ")
            if strs[0] =="f":
                m=m+1
                face[i].append((int(strs[1]), int(strs[2]), int(strs[3])))
            line = file.readline()
            strs = line.split(" ")
        if strs[0] =="f":
            m=m+1
            face[i].append((int(strs[1]), int(strs[2]), int(strs[3])))
        if strs[0] == "v":
            n=n+1
            points[j].append((float(strs[2]), float(strs[3]), float(strs[4])))
            ppoints.append((float(strs[2]), float(strs[3]), float(strs[4])))

face_to_face=[[] for i in range(8)]
ff=i
#print(ff)
i=0
a=0
while i<8:#ff=8
    l=0
    sum=0
    for x in range(0,i+1):
        sum=sum+len(points[x])
    #print("sum=",sum)
    for j in range(len(points[i])):
        for k in range(len(points[i+1])):
            if(points[i][j]==points[i+1][k]):
                #print(points[i][j],points[i+1][k],j,k)
                face_to_face[i].append((j+sum-len(points[i]),k+sum))
                l=l+1
    i=i+1

def Area(p1,p2,p3):
    a=math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)
    b=math.sqrt((p3[0]-p2[0])**2+(p3[1]-p2[1])**2+(p3[2]-p2[2])**2)
    c=math.sqrt((p3[0]-p1[0])**2+(p3[1]-p1[1])**2+(p3[2]-p1[2])**2)
    p=(a+b+c)/2
    S=math.sqrt(p*(p-a)*(p-b)*(p-c))
    return S

S=[0]*9
for j in range(9):
    for i in range(len(face[j])):
    #print(face[0][i])
        S[j]=S[j]+Area(ppoints[(face[j][i][0])-1],ppoints[(face[j][i][1])-1],ppoints[(face[j][i][2])-1])
#print(S)
S[5]=S[3]
S[6]=S[2]
S[7]=S[1]
S[8]=S[0]
#print(S)


SS=[0]*8
ff=[[] for i in range(500)]
for i in range(len(face[0])):
    #print(i)
    if(face[0][i][0]<=65 and face[0][i][1]<=65 and face[0][i][2]<=65):
        ff[0].append(face[0][i])
#print((ff[0]))
#print(len(points[0]))
for i in range(len(face[1])):
    if(face[1][i][0]<=65+len(points[0]) and face[1][i][1]<=65+len(points[0]) and face[1][i][2]<=65+len(points[0])):
        ff[1].append(face[1][i])
for i in range(len(face[2])):
    if(face[2][i][0]>-65+len(points[0])+len(points[1])+len(points[2]) and face[2][i][1]>-65+len(points[0])+len(points[1])+len(points[2]) and face[2][i][2]>-65+len(points[0])+len(points[1])+len(points[2])):
        ff[2].append(face[2][i])
for i in range(len(face[3])):
    if(face[3][i][0]<=65+len(points[0])+len(points[1])+len(points[2]) and face[3][i][1]<=65+len(points[0])+len(points[1])+len(points[2]) and face[3][i][2]<=65+len(points[0])+len(points[1])+len(points[2])):
        ff[3].append(face[3][i])

for j in range(4):
    for i in range(len(ff[j])):
    #print(face[0][i])
        SS[j]=SS[j]+Area(ppoints[(ff[j][i][0])-1],ppoints[(ff[j][i][1])-1],ppoints[(ff[j][i][2])-1])
SS[4]=SS[3]
SS[5]=SS[2]
SS[6]=SS[1]
SS[7]=SS[0]
#print(S)
#print(SS)#S_{ij}
S[0]=S[0]-SS[0]
S[8]=S[8]-SS[7]
for i in range(1,8):
    #print(i)
    S[i]=S[i]-SS[i-1]-SS[i]
#print(S)#Si

#print((ff[2]))
#print(len(ff[2]))
#print(ppoints[516],ppoints[518],ppoints[533])

A=1
eps = [0.05,0.05,0.05,0.02,0.1,0.01,0.05,0.05,0.05]
c = [900,900,900,1930,520,840,900,900,900]
QR=[0,"A*(22+2*sympy.sin(t/8))",0,0,0,0,0,"A*(22+2*sympy.sin(t/6))",0]
QRA=[0,1,0,0,0,0,0,1,0]
QRS=[0,1/8,0,0,0,0,0,1/8,0]
Lambda=[240,240,118,9.7,10.5,119,240,240,0]

dataframe = pd.DataFrame({'eps':eps,'c':c,'QR':QR,'Lambda':Lambda,'QRA':QRA,'QRS':QRS})

dataframe.to_csv("coefficients.csv",index=False,sep=',')

data = pd.read_csv('coefficients.csv')
kij=[0]*len(SS)
for i in range(len(SS)):
    kij[i]=data[['Lambda']].iloc[i,0]*SS[i]
print(kij)
QiR=[0]*len(S)
QiRA=[0]*len(S)
QiRS=[0]*len(S)
epsi=[0]*len(S)
ci=[0]*len(S)
for i in range(len(S)):
    #QiR[i]=eval(data[['QR']].iloc[i,0])
    QiRA[i]=data[['QRA']].iloc[i,0]
    QiRS[i]=data[['QRS']].iloc[i,0]
    epsi[i]=data[['eps']].iloc[i,0]
    ci[i]=data[['c']].iloc[i,0]
# print(QiRA)
# print(QiRS)
# print(epsi)
# print(ci)

def sys_of_funcs(T, t, ci,kij,epsi,S,QiRA,QiRS):
    c0=5.67
    y1 = T[0]
    y2 = T[1]
    y3 = T[2]
    y4 = T[3]
    y5 = T[4]
    y6 = T[5]
    y7 = T[6]
    y8 = T[7]
    y9 = T[8]
    f1 = (1/ci[0])*(-kij[0]*(y2-y1)-epsi[0]*S[0]*c0*((y1/100)**4))
    f2 = (1/ci[1])*(-kij[0]*(y2-y1)-kij[1]*(y3-y2)-epsi[1]*S[1]*c0*((y2/100)**4)+QiRA[1]*(22+2*math.sin(t*QiRS[1])))
    f3 = (1/ci[2])*(-kij[1]*(y3-y2)-kij[2]*(y4-y3)-epsi[2]*S[2]*c0*((y3/100)**4))
    f4 = (1/ci[3])*(-kij[2]*(y4-y3)-kij[3]*(y5-y4)-epsi[3]*S[3]*c0*((y4/100)**4))
    f5 = (1/ci[4])*(-kij[3]*(y5-y4)-kij[4]*(y6-y5)-epsi[4]*S[4]*c0*((y5/100)**4))
    f6 = (1/ci[5])*(-kij[4]*(y6-y5)-kij[5]*(y7-y6)-epsi[5]*S[5]*c0*((y6/100)**4))
    f7 = (1/ci[6])*(-kij[5]*(y7-y6)-kij[6]*(y8-y7)-epsi[6]*S[6]*c0*((y7/100)**4))
    f8 = (1/ci[7])*(-kij[6]*(y8-y7)-kij[7]*(y9-y8)-epsi[7]*S[7]*c0*((y8/100)**4)+QiRA[7]*(22+2*math.sin(t*QiRS[7])))
    f9 = (1/ci[8])*(-kij[7]*(y9-y8)-epsi[8]*S[8]*c0*((y9/100)**4))
    return [f1, f2,f3,f4,f5,f6,f7,f8,f9]

#init = 20.0,20.0 ,20.0,20.0,20.0,20.0,20.0,20.0,20.0       # (3.041592653589793, 0.0)
#init=20,10,10,20,23,20,0,30,50
init=20,10,10,20,23,20,0,30,50
t = np.linspace(0,20, 1001)
print(t)
#print(QiR,S)
sol = odeint(sys_of_funcs, init, t, args=(ci,kij,epsi,S,QiRA,QiRS))
#print(sol)
plt.grid()
plt.xlabel('t')
plt.ylabel('T')
plt.plot(t, sol[:, 0], color='b', label=r"$T_1$")
plt.plot(t, sol[:, 1], color='c', label=r"$T_2$")
plt.plot(t, sol[:, 2], color='r', label=r"$T_3$")
plt.plot(t, sol[:, 3], color='g', label=r"$T_4$")
plt.plot(t, sol[:, 4], color='m', label=r"$T_5$")
plt.plot(t, sol[:, 5], color='pink', label=r"$T_6$")
plt.plot(t, sol[:, 6], color='cyan', label=r"$T_7$")
plt.plot(t, sol[:, 7], color='tomato', label=r"$T_8$")
plt.plot(t, sol[:, 8], color='orangered', label=r"$T_9$")

plt.legend(loc="best")
plt.show()

dataframe2 = pd.DataFrame({'t':t,'T1':sol[:, 0],'T2':sol[:, 1],'T3':sol[:, 2],'T4':sol[:, 3],'T5':sol[:, 4],'T6':sol[:, 5],'T7':sol[:, 6],'T8':sol[:, 7],'T9':sol[:, 8]})
dataframe2.to_csv("Results.csv",index=False,sep=',')



IS_PERSPECTIVE = True  # 透视投影
VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 20.0])  # 视景体的left/right/bottom/top/near/far六个面
SCALE_K = np.array([0.5, 0.5, 0.5])  # 模型缩放比例
EYE = np.array([0.0, 10.0, 20.0])  # 眼睛的位置（默认z轴的正方向）
LOOK_AT = np.array([0.0, 10.0, 0.0])  # 瞄准方向的参考点（默认在坐标原点）
EYE_UP = np.array([0.0, 0.0, 10.0])  # 定义对观察者而言的上方（默认y轴的正方向）
WIN_W, WIN_H = 640, 480  # 保存窗口宽度和高度的变量
LEFT_IS_DOWNED = False  # 鼠标左键被按下
MOUSE_X, MOUSE_Y = 0, 0  # 考察鼠标位移量时保存的起始位置

with open('Results.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['t'] == '2.6':
            # print(type(row))
            ser = Series(row)
            # print('ser\n', ser)
            # print(ser)
            arr = np.array(ser)
            arr = arr.astype(float)
            # print(type((arr)))
arr = arr[1:]
print(arr)

def getposture():
    global EYE, LOOK_AT

    dist = np.sqrt(np.power((EYE - LOOK_AT), 2).sum())
    if dist > 0:
        phi = np.arcsin((EYE[1] - LOOK_AT[1]) / dist)
        theta = np.arcsin((EYE[0] - LOOK_AT[0]) / (dist * np.cos(phi)))
    else:
        phi = 0.0
        theta = 0.0

    return dist, phi, theta


DIST, PHI, THETA = getposture()  # 眼睛与观察目标之间的距离、仰角、方位角


def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)  # 设置画布背景色。注意：这里必须是4个参数
    glEnable(GL_DEPTH_TEST)  # 开启深度测试，实现遮挡关系
    glDepthFunc(GL_LEQUAL)  # 设置深度测试函数（GL_LEQUAL只是选项之一）


def draw():
    global IS_PERSPECTIVE, VIEW
    global EYE, LOOK_AT, EYE_UP
    global SCALE_K
    global WIN_W, WIN_H

    # 清除屏幕及深度缓存
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # 设置投影（透视投影）
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    if WIN_W > WIN_H:
        if IS_PERSPECTIVE:
            glFrustum(VIEW[0] * WIN_W / WIN_H, VIEW[1] * WIN_W / WIN_H, VIEW[2], VIEW[3], VIEW[4], VIEW[5])
        else:
            glOrtho(VIEW[0] * WIN_W / WIN_H, VIEW[1] * WIN_W / WIN_H, VIEW[2], VIEW[3], VIEW[4], VIEW[5])
    else:
        if IS_PERSPECTIVE:
            glFrustum(VIEW[0], VIEW[1], VIEW[2] * WIN_H / WIN_W, VIEW[3] * WIN_H / WIN_W, VIEW[4], VIEW[5])
        else:
            glOrtho(VIEW[0], VIEW[1], VIEW[2] * WIN_H / WIN_W, VIEW[3] * WIN_H / WIN_W, VIEW[4], VIEW[5])

    # 设置模型视图
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # 几何变换
    glScale(SCALE_K[0], SCALE_K[1], SCALE_K[2])

    # 设置视点
    gluLookAt(
        EYE[0], EYE[1], EYE[2],
        LOOK_AT[0], LOOK_AT[1], LOOK_AT[2],
        EYE_UP[0], EYE_UP[1], EYE_UP[2]
    )

    # 设置视口
    glViewport(0, 0, WIN_W, WIN_H)

    # ---------------------------------------------------------------
    glBegin(GL_LINES)  # 开始绘制线段（世界坐标系）

    # 以红色绘制x轴
    glColor4f(1.0, 0.0, 0.0, 1.0)  # 设置当前颜色为红色不透明
    glVertex3f(-10, 0.0, 0.0)  # 设置x轴顶点（x轴负方向）
    glVertex3f(10, 0.0, 0.0)  # 设置x轴顶点（x轴正方向）

    # 以绿色绘制y轴
    glColor4f(0.0, 1.0, 0.0, 1.0)  # 设置当前颜色为绿色不透明
    glVertex3f(0.0, -5, 0.0)  # 设置y轴顶点（y轴负方向）
    glVertex3f(0.0, 30, 0.0)  # 设置y轴顶点（y轴正方向）

    # 以蓝色绘制z轴
    glColor4f(0.0, 0.0, 1.0, 1.0)  # 设置当前颜色为蓝色不透明
    glVertex3f(0.0, 0.0, -10)  # 设置z轴顶点（z轴负方向）
    glVertex3f(0.0, 0.0, 10)  # 设置z轴顶点（z轴正方向）

    glEnd()  # 结束绘制线段

    # ---------------------------------------------------------------
    objFilePath = 'model3.obj'
    with open(objFilePath) as file:
        points = [[] for i in range(9)]
        face = [[] for i in range(9)]
        ppoints = []
        n = 0
        m = 0
        i = -1
        j = 0
        while 1:
            line = file.readline()
            strs = line.split(" ")
            if not line:
                break
            if strs[0] == "g":
                i = i + 1
                j = j + 1
                line = file.readline()
                strs = line.split(" ")
                if strs[0] == "f":
                    m = m + 1
                    face[i].append((int(strs[1]), int(strs[2]), int(strs[3])))
                line = file.readline()
                strs = line.split(" ")
            if strs[0] == "f":
                m = m + 1
                face[i].append((int(strs[1]), int(strs[2]), int(strs[3])))
            if strs[0] == "v":
                n = n + 1
                points[j].append((float(strs[2]), float(strs[3]), float(strs[4])))
                ppoints.append((float(strs[2]), float(strs[3]), float(strs[4])))

    Max=max(arr)
    Min=min(arr)
    #points = np.array(points)
    for i in range(len(face[0])):
        glBegin(GL_TRIANGLES)  # 开始绘制三角形（z轴负半区）
        if(arr[0]>(Max+Min)/2):
            R=(arr[0]-((Max+Min)/2))*0.5/((Max-Min)/4)
            G=(Max-arr[0])*0.5/((Max-Min)/4)
            B=0
        else:
            R=0
            G=(arr[0]-Min)*0.5/((Max-Min)/4)
            B=(((Max+Min)/2)-arr[0])*0.5/((Max-Min)/4)
        glColor4f(R, G, B, 0.5)  # 设置当前颜色为红色不透明
        glVertex3f(points[0][(face[0][i][0]-1)][0], points[0][(face[0][i][0]-1)][1], points[0][(face[0][i][0]-1)][2])  # 设置三角形顶点
        glColor4f(R, G, B, 0.5)  # 设置当前颜色为绿色不透明
        glVertex3f(points[0][(face[0][i][1]-1)][0], points[0][(face[0][i][1]-1)][1], points[0][(face[0][i][1]-1)][2])  # 设置三角形顶点
        glColor4f(R, G, B, 0.5)  # 设置当前颜色为蓝色不透明
        glVertex3f(points[0][(face[0][i][2]-1)][0], points[0][(face[0][i][2]-1)][1], points[0][(face[0][i][2]-1)][2])  # 设置三角形顶点
        #print(points[0])
        glEnd()  # 结束绘制三角形
    for i in range(len(face[1])):
        glBegin(GL_TRIANGLES)  # 开始绘制三角形（z轴负半区）
        if (arr[1] > (Max + Min) / 2):
            R = (arr[1] - ((Max + Min) / 2)) * 0.5 / ((Max - Min) / 4)
            G = (Max - arr[1]) * 0.5 / ((Max - Min) / 4)
            B = 0
        else:
            R = 0
            G = (arr[1] - Min) * 0.5 / ((Max - Min) / 4)
            B = (((Max + Min) / 2) - arr[1]) * 0.5 / ((Max - Min) / 4)
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为红色不透明
        glVertex3f(points[1][(face[1][i][0]-1-len(points[0]))][0], points[1][(face[1][i][0]-1-len(points[0]))][1], points[1][(face[1][i][0]-1-len(points[0]))][2])  # 设置三角形顶点
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为绿色不透明
        glVertex3f(points[1][(face[1][i][1]-1-len(points[0]))][0], points[1][(face[1][i][1]-1-len(points[0]))][1], points[1][(face[1][i][1]-1-len(points[0]))][2])  # 设置三角形顶点
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为蓝色不透明
        glVertex3f(points[1][(face[1][i][2]-1-len(points[0]))][0], points[1][(face[1][i][2]-1-len(points[0]))][1], points[1][(face[1][i][2]-1-len(points[0]))][2])  # 设置三角形顶点

        glEnd()  # 结束绘制三角形
    for i in range(len(face[2])):
        glBegin(GL_TRIANGLES)  # 开始绘制三角形（z轴负半区）
        if (arr[2] > (Max + Min) / 2):
            R = (arr[2] - ((Max + Min) / 2)) * 0.5 / ((Max - Min) / 4)
            G = (Max - arr[2]) * 0.5 / ((Max - Min) / 4)
            B = 0
        else:
            R = 0
            G = (arr[2] - Min) * 0.5 / ((Max - Min) / 4)
            B = (((Max + Min) / 2) - arr[2]) * 0.5 / ((Max - Min) / 4)
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为红色不透明
        glVertex3f(points[2][(face[2][i][0]-1-len(points[0])-len(points[1]))][0], points[2][(face[2][i][0]-1-len(points[0])-len(points[1]))][1], points[2][(face[2][i][0]-1-len(points[0])-len(points[1]))][2])  # 设置三角形顶点
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为绿色不透明
        glVertex3f(points[2][(face[2][i][1]-1-len(points[0])-len(points[1]))][0], points[2][(face[2][i][1]-1-len(points[0])-len(points[1]))][1], points[2][(face[2][i][1]-1-len(points[0])-len(points[1]))][2])  # 设置三角形顶点
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为蓝色不透明
        glVertex3f(points[2][(face[2][i][2]-1-len(points[0])-len(points[1]))][0], points[2][(face[2][i][2]-1-len(points[0])-len(points[1]))][1], points[2][(face[2][i][2]-1-len(points[0])-len(points[1]))][2])  # 设置三角形顶点
        glEnd()  # 结束绘制三角形
    for i in range(len(face[3])):
        glBegin(GL_TRIANGLES)  # 开始绘制三角形（z轴负半区）
        if (arr[3] > (Max + Min) / 2):
            R = (arr[3] - ((Max + Min) / 2)) * 0.5 / ((Max - Min) / 4)
            G = (Max - arr[3]) * 0.5 / ((Max - Min) / 4)
            B = 0
        else:
            R = 0
            G = (arr[3] - Min) * 0.5 / ((Max - Min) / 4)
            B = (((Max + Min) / 2) - arr[3]) * 0.5 / ((Max - Min) / 4)
        sum=len(points[0])+len(points[1])+len(points[2])
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为红色不透明
        glVertex3f(points[3][(face[3][i][0]-1-sum)][0], points[3][(face[3][i][0]-1-sum)][1], points[3][(face[3][i][0]-1-sum)][2])  # 设置三角形顶点
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为绿色不透明
        glVertex3f(points[3][(face[3][i][1]-1-sum)][0], points[3][(face[3][i][1]-1-sum)][1], points[3][(face[3][i][1]-1-sum)][2])  # 设置三角形顶点
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为蓝色不透明
        glVertex3f(points[3][(face[3][i][2]-1-sum)][0], points[3][(face[3][i][2]-1-sum)][1], points[3][(face[3][i][2]-1-sum)][2])  # 设置三角形顶点
        glEnd()  # 结束绘制三角形
    for i in range(len(face[4])):
        glBegin(GL_TRIANGLES)  # 开始绘制三角形（z轴负半区）
        if (arr[4] > (Max + Min) / 2):
            R = (arr[4] - ((Max + Min) / 2)) * 0.5 / ((Max - Min) / 4)
            G = (Max - arr[4]) * 0.5 / ((Max - Min) / 4)
            B = 0
        else:
            R = 0
            G = (arr[4] - Min) * 0.5 / ((Max - Min) / 4)
            B = (((Max + Min) / 2) - arr[4]) * 0.5 / ((Max - Min) / 4)

        sum=len(points[0])+len(points[1])+len(points[2])+len(points[3])
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为红色不透明
        glVertex3f(points[4][(face[4][i][0]-1-sum)][0], points[4][(face[4][i][0]-1-sum)][1], points[4][(face[4][i][0]-1-sum)][2])  # 设置三角形顶点
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为绿色不透明
        glVertex3f(points[4][(face[4][i][1]-1-sum)][0], points[4][(face[4][i][1]-1-sum)][1], points[4][(face[4][i][1]-1-sum)][2])  # 设置三角形顶点
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为蓝色不透明
        glVertex3f(points[4][(face[4][i][2]-1-sum)][0], points[4][(face[4][i][2]-1-sum)][1], points[4][(face[4][i][2]-1-sum)][2])  # 设置三角形顶点
        glEnd()  # 结束绘制三角形
    for i in range(len(face[5])):
        glBegin(GL_TRIANGLES)  # 开始绘制三角形（z轴负半区）
        if (arr[7] > (Max + Min) / 2):
            R = (arr[7] - ((Max + Min) / 2)) * 0.5 / ((Max - Min) / 4)
            G = (Max - arr[7]) * 0.5 / ((Max - Min) / 4)
            B = 0
        else:
            R = 0
            G = (arr[7] - Min) * 0.5 / ((Max - Min) / 4)
            B = (((Max + Min) / 2) - arr[7]) * 0.5 / ((Max - Min) / 4)
        sum=len(points[0])+len(points[1])+len(points[2])+len(points[3])+len(points[4])
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为红色不透明
        glVertex3f(points[5][(face[5][i][0]-1-sum)][0], points[5][(face[5][i][0]-1-sum)][1], points[5][(face[5][i][0]-1-sum)][2])  # 设置三角形顶点
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为绿色不透明
        glVertex3f(points[5][(face[5][i][1]-1-sum)][0], points[5][(face[5][i][1]-1-sum)][1], points[5][(face[5][i][1]-1-sum)][2])  # 设置三角形顶点
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为蓝色不透明
        glVertex3f(points[5][(face[5][i][2]-1-sum)][0], points[5][(face[5][i][2]-1-sum)][1], points[5][(face[5][i][2]-1-sum)][2])  # 设置三角形顶点
        glEnd()  # 结束绘制三角形
    for i in range(len(face[6])):
        glBegin(GL_TRIANGLES)  # 开始绘制三角形（z轴负半区）
        if (arr[6] > (Max + Min) / 2):
            R = (arr[6] - ((Max + Min) / 2)) * 0.5 / ((Max - Min) / 4)
            G = (Max - arr[6]) * 0.5 / ((Max - Min) / 4)
            B = 0
        else:
            R = 0
            G = (arr[6] - Min) * 0.5 / ((Max - Min) / 4)
            B = (((Max + Min) / 2) - arr[6]) * 0.5 / ((Max - Min) / 4)
        sum=len(points[0])+len(points[1])+len(points[2])+len(points[3])+len(points[4])+len(points[5])
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为红色不透明
        glVertex3f(points[6][(face[6][i][0]-1-sum)][0], points[6][(face[6][i][0]-1-sum)][1], points[6][(face[6][i][0]-1-sum)][2])  # 设置三角形顶点
        glColor4f(R,G,B, 0.5)   # 设置当前颜色为绿色不透明
        glVertex3f(points[6][(face[6][i][1]-1-sum)][0], points[6][(face[6][i][1]-1-sum)][1], points[6][(face[6][i][1]-1-sum)][2])  # 设置三角形顶点
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为蓝色不透明
        glVertex3f(points[6][(face[6][i][2]-1-sum)][0], points[6][(face[6][i][2]-1-sum)][1], points[6][(face[6][i][2]-1-sum)][2])  # 设置三角形顶点
        glEnd()  # 结束绘制三角形
    for i in range(len(face[7])):
        glBegin(GL_TRIANGLES)  # 开始绘制三角形（z轴负半区）
        if (arr[5] > (Max + Min) / 2):
            R = (arr[5] - ((Max + Min) / 2)) * 0.5 / ((Max - Min) / 4)
            G = (Max - arr[5]) * 0.5 / ((Max - Min) / 4)
            B = 0
        else:
            R = 0
            G = (arr[5] - Min) * 0.5 / ((Max - Min) / 4)
            B = (((Max + Min) / 2) - arr[5]) * 0.5 / ((Max - Min) / 4)
        sum=len(points[0])+len(points[1])+len(points[2])+len(points[3])+len(points[4])+len(points[5])+len(points[6])
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为红色不透明
        glVertex3f(points[7][(face[7][i][0]-1-sum)][0], points[7][(face[7][i][0]-1-sum)][1], points[7][(face[7][i][0]-1-sum)][2])  # 设置三角形顶点
        glColor4f(R,G,B, 0.5)   # 设置当前颜色为绿色不透明
        glVertex3f(points[7][(face[7][i][1]-1-sum)][0], points[7][(face[7][i][1]-1-sum)][1], points[7][(face[7][i][1]-1-sum)][2])  # 设置三角形顶点
        glColor4f(R,G,B, 0.5)   # 设置当前颜色为蓝色不透明
        glVertex3f(points[7][(face[7][i][2]-1-sum)][0], points[7][(face[7][i][2]-1-sum)][1], points[7][(face[7][i][2]-1-sum)][2])  # 设置三角形顶点
        glEnd()  # 结束绘制三角形
    for i in range(len(face[8])):
        glBegin(GL_TRIANGLES)  # 开始绘制三角形（z轴负半区）
        if (arr[8] > (Max + Min) / 2):
            R = (arr[8] - ((Max + Min) / 2)) * 0.5 / ((Max - Min) / 4)
            G = (Max - arr[8]) * 0.5 / ((Max - Min) / 4)
            B = 0
        else:
            R = 0
            G = (arr[8] - Min) * 0.5 / ((Max - Min) / 4)
            B = (((Max + Min) / 2) - arr[8]) * 0.5 / ((Max - Min) / 4)
        sum=len(points[0])+len(points[1])+len(points[2])+len(points[3])+len(points[4])+len(points[5])+len(points[6])+len(points[7])
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为红色不透明
        glVertex3f(points[8][(face[8][i][0]-1-sum)][0], points[8][(face[8][i][0]-1-sum)][1], points[8][(face[8][i][0]-1-sum)][2])  # 设置三角形顶点
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为绿色不透明
        glVertex3f(points[8][(face[8][i][1]-1-sum)][0], points[8][(face[8][i][1]-1-sum)][1], points[8][(face[8][i][1]-1-sum)][2])  # 设置三角形顶点
        glColor4f(R,G,B, 0.5)  # 设置当前颜色为蓝色不透明
        glVertex3f(points[8][(face[8][i][2]-1-sum)][0], points[8][(face[8][i][2]-1-sum)][1], points[8][(face[8][i][2]-1-sum)][2])  # 设置三角形顶点
        glEnd()  # 结束绘制三角形
    glutSwapBuffers()  # 切换缓冲区，以显示绘制内容


def reshape(width, height):
    global WIN_W, WIN_H

    WIN_W, WIN_H = width, height
    glutPostRedisplay()


def mouseclick(button, state, x, y):
    global SCALE_K
    global LEFT_IS_DOWNED
    global MOUSE_X, MOUSE_Y

    MOUSE_X, MOUSE_Y = x, y
    if button == GLUT_LEFT_BUTTON:
        LEFT_IS_DOWNED = state == GLUT_DOWN
    elif button == 3:
        SCALE_K *= 1.05
        glutPostRedisplay()
    elif button == 4:
        SCALE_K *= 0.95
        glutPostRedisplay()


def mousemotion(x, y):
    global LEFT_IS_DOWNED
    global EYE, EYE_UP
    global MOUSE_X, MOUSE_Y
    global DIST, PHI, THETA
    global WIN_W, WIN_H

    if LEFT_IS_DOWNED:
        dx = MOUSE_X - x
        dy = y - MOUSE_Y
        MOUSE_X, MOUSE_Y = x, y

        PHI += 2 * np.pi * dy / WIN_H
        PHI %= 2 * np.pi
        THETA += 2 * np.pi * dx / WIN_W
        THETA %= 2 * np.pi
        r = DIST * np.cos(PHI)

        EYE[1] = DIST * np.sin(PHI)
        EYE[0] = r * np.sin(THETA)
        EYE[2] = r * np.cos(THETA)

        if 0.5 * np.pi < PHI < 1.5 * np.pi:
            EYE_UP[1] = -1.0
        else:
            EYE_UP[1] = 1.0

        glutPostRedisplay()


def keydown(key, x, y):
    global DIST, PHI, THETA
    global EYE, LOOK_AT, EYE_UP
    global IS_PERSPECTIVE, VIEW

    if key in [b'x', b'X', b'y', b'Y', b'z', b'Z']:
        if key == b'x':  # 瞄准参考点 x 减小
            LOOK_AT[0] -= 0.01
        elif key == b'X':  # 瞄准参考 x 增大
            LOOK_AT[0] += 0.01
        elif key == b'y':  # 瞄准参考点 y 减小
            LOOK_AT[1] -= 0.01
        elif key == b'Y':  # 瞄准参考点 y 增大
            LOOK_AT[1] += 0.01
        elif key == b'z':  # 瞄准参考点 z 减小
            LOOK_AT[2] -= 0.01
        elif key == b'Z':  # 瞄准参考点 z 增大
            LOOK_AT[2] += 0.01

        DIST, PHI, THETA = getposture()
        glutPostRedisplay()
    elif key == b'\r':  # 回车键，视点前进
        EYE = LOOK_AT + (EYE - LOOK_AT) * 0.9
        DIST, PHI, THETA = getposture()
        glutPostRedisplay()
    elif key == b'\x08':  # 退格键，视点后退
        EYE = LOOK_AT + (EYE - LOOK_AT) * 1.1
        DIST, PHI, THETA = getposture()
        glutPostRedisplay()
    elif key == b' ':  # 空格键，切换投影模式
        IS_PERSPECTIVE = not IS_PERSPECTIVE
        glutPostRedisplay()



glutInit()
displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH
glutInitDisplayMode(displayMode)

glutInitWindowSize(WIN_W, WIN_H)
glutInitWindowPosition(300, 200)
glutCreateWindow('Result')

init()  # 初始化画布
glutDisplayFunc(draw)  # 注册回调函数draw()
glutReshapeFunc(reshape)  # 注册响应窗口改变的函数reshape()
glutMouseFunc(mouseclick)  # 注册响应鼠标点击的函数mouseclick()
glutMotionFunc(mousemotion)  # 注册响应鼠标拖拽的函数mousemotion()
glutKeyboardFunc(keydown)  # 注册键盘输入的函数keydown()

glutMainLoop()  # 进入glut主循环
