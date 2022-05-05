import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
import time
from scipy.integrate import odeint
import pandas as pd

start=time.time()
au,G,RE,ME = 1.48e11,6.67e-11,1.48e11,5.965e24
m = np.array([3.32e5,0.055,0.815,1,
              0.107,317.8,95.16,14.54,17.14])*ME*6.67e-11
r = np.array([0,0.387,0.723,1,1.524,5.203,
              9.537,19.19,30.7])*RE

#theta = np.random.rand(9)*np.pi*2
theta=[0.90579977,4.76568695, 1.34869972, 6.02969388, 2.24714959, 3.45095948,
 3.41281759, 4.32174632, 2.33019222]
x = r*np.cos(theta)
y = r*np.sin(theta)
v = np.array([0,47.89,35.03,29.79,
              24.13,13.06,9.64,6.81,5.43])*1000
v_x = -v*np.sin(theta)
v_y = v*np.cos(theta)
#速度
# v_x = np.array([0.0,0,0])
# v_y = np.array([0,2.88e4,2.4e4])

# N = 1000
# dt = 36000
# ts =  np.arange(0,N*dt,dt)/3600/24 #创造等差数列
# xs,ys = [],[] #记录位置

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(xlim=(-31*RE,31*RE),ylim=(-31*RE,31*RE))
ax.grid()
traces = [ax.plot([],[],'-', lw=0.5)[0] for _ in range(9)]
pts = [ax.plot([],[],marker='o')[0] for _ in range(9)]
k_text = ax.text(0.05,0.85,'',transform=ax.transAxes)
textTemplate = 't = %.3f days\n'

N = 2
dt = 360000
ts =  np.arange(0,N*dt,dt)
xs,ys = [],[]

def sys_of_funcs(init,t,x_ij,r_ij,m,i):
    r0,v0=init
    f=0
    for j in range(len(m)):
        if i != j:
            f+=(m[j]*x_ij[i,j]/r_ij[i,j]**3)
    return np.array([v0,f])
#print(x,y)
x_ij = (x-x.reshape(len(m),1))
y_ij = (y-y.reshape(len(m),1))
r_ij = np.sqrt(x_ij**2+y_ij**2)
#print(r_ij)
plt.grid()
xx=np.zeros(len(m))
yy=np.zeros(len(m))
for i in range(len(m)):
    result = odeint(sys_of_funcs, (x[i],v_x[i]), ts, args=(x_ij, r_ij, m, i))
    result2 = odeint(sys_of_funcs, (y[i],v_y[i]), ts, args=(y_ij, r_ij, m, i))
    plt.plot(result[:,0],result2[:,0])
    x[i]=result[-1,0]
    y[i]=result2[-1,0]
    v_x[i]=result[-1,-1]
    v_y[i]=result2[-1,-1]
    #print(len(result),len(result[0]))
    #print("res",result[-1],"res2",result2)
# print("x,y",x,y)
# print("v_x,v_y",v_x,v_y)

# x=xx
# y=yy
# x_ij = (x-x.reshape(len(m),1))
# y_ij = (y-y.reshape(len(m),1))
# r_ij = np.sqrt(x_ij**2+y_ij**2)
# #print(r_ij)
# plt.grid()
# xx=np.zeros(len(m))
# yy=np.zeros(len(m))
# for i in range(len(m)):
#     result = odeint(sys_of_funcs, (x[i],v_x[i]), ts, args=(x_ij, r_ij, m, i))
#     result2 = odeint(sys_of_funcs, (y[i],v_y[i]), ts, args=(y_ij, r_ij, m, i))
#     plt.plot(result[:,0],result2[:,0])
#     xx[i]=result[-1,0]
#     yy[i]=result2[-1,0]
#     v_x[i] = result[-1, -1]
#     v_y[i] = result2[-1, -1]
xs.append(x.tolist())
ys.append(y.tolist())
tts =  np.arange(0,250*N*dt,dt)
for _ in tts:
    x_ij = (x - x.reshape(len(m), 1))
    y_ij = (y - y.reshape(len(m), 1))
    r_ij = np.sqrt(x_ij ** 2 + y_ij ** 2)
    for i in range(len(m)):
        result = odeint(sys_of_funcs, (x[i], v_x[i]), ts, args=(x_ij, r_ij, m, i))
        result2 = odeint(sys_of_funcs, (y[i], v_y[i]), ts, args=(y_ij, r_ij, m, i))
        #plt.plot(result[:,0],result2[:,0])
        #print(result[:, 0], result2[:, 0])
        x[i]=result[-1,0]
        y[i]=result2[-1,0]
        v_x[i] = result[-1, -1]
        v_y[i] = result2[-1, -1]
    xs.append(x.tolist())
    ys.append(y.tolist())
xs = np.array(xs)
ys = np.array(ys)
#plt.show()
#print(len(xs),len(xs[0]))

# for _ in ts:
#     x_ij = (x - x.reshape(len(m), 1))
#     y_ij = (y - y.reshape(len(m), 1))
#     r_ij = np.sqrt(x_ij ** 2 + y_ij ** 2)
#     for i in range(len(m)):
#         for j in range(len(m)):
#             if i!=j :
#                 accx_1[i] += (m[j]*x_ij[i,j]/r_ij[i,j]**3)
#                 accy_1[i] += (m[j]*y_ij[i,j]/r_ij[i,j]**3)
#                 #print(accx_1[i], accy_1[i])
#     v_x += 0.5*(accx_0+accx_1)*dt
#     v_y += 0.5*(accy_0+accy_1)*dt
#     x += v_x * dt + 0.5 * accx_1 * dt ** 2
#     y += v_y * dt + 0.5 * accy_1 * dt ** 2
#     #print(x,y)
#     xs.append(x.tolist())
#     ys.append(y.tolist())
#     accx_0 = accx_1
#     accy_0 = accy_1
#     accx_1 = np.zeros(len(m))
#     accy_1 = np.zeros(len(m))
# xs = np.array(xs)
# ys = np.array(ys)


def animate(n):
    for i in range(len(m)):
        traces[i].set_data(xs[:n,i],ys[:n,i])
        pts[i].set_data(xs[n,i],ys[n,i])
    #k_text.set_text(textTemplate % (ts[n]/3600/24))
    return traces+pts+[k_text]
end=time.time()
print(end-start)
dataframe = pd.DataFrame({'x1': xs[:,0], 'y1': ys[:,0],'x2': xs[:,1], 'y2': ys[:,1],'x3': xs[:,2], 'y3': ys[:,2],
                              'x4': xs[:,3], 'y4': ys[:,3],'x5': xs[:,4], 'y5': ys[:,4],'x6': xs[:,5], 'y6': ys[:,5],
                              'x7': xs[:,6], 'y7': ys[:,6],'x8': xs[:,7], 'y8': ys[:,7],'x9': xs[:,8], 'y9': ys[:,8]})
dataframe.to_csv("ode.csv", index=False, sep=',')
#
N=1000
ani = FuncAnimation(fig, animate,
    range(N), interval=100, blit=True)
plt.show()
# ani.save("solar4ok.gif",writer='pillow')
# ani.save("solar4.mp4",writer='ffmpeg',fps=1000/50)


