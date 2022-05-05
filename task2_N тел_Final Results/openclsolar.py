#!/usr/bin/env python
import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
import time
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import pandas as pd
nn = 9
N = 500
dt = 360000
au, G, RE, ME = 1.48e11, 6.67e-11, 1.48e11, 5.965e24
m = np.array([3.32e5,0.055,0.815,1,
              0.107,317.8,95.16,14.54,17.14])*ME*6.67e-11

r = np.array([0,0.387,0.723,1,1.524,5.203,
              9.537,19.19,30.7])*RE
# m = np.random.rand(nn) * 1000 * ME * 6.67e-11
# r = np.random.rand(nn) * 10 * RE
#theta = np.random.rand(nn) * np.pi * 2
theta=[0.90579977,4.76568695, 1.34869972, 6.02969388, 2.24714959, 3.45095948,
 3.41281759, 4.32174632, 2.33019222]
x = r*np.cos(theta)
y = r*np.sin(theta)
v = np.array([0,47.89,35.03,29.79,
              24.13,13.06,9.64,6.81,5.43])*1000
v_x = -v*np.sin(theta)
v_y = v*np.cos(theta)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(xlim=(-31*RE,31*RE),ylim=(-31*RE,31*RE))
ax.grid()
traces = [ax.plot([],[],'-', lw=0.5)[0] for _ in range(9)]
pts = [ax.plot([],[],marker='o')[0] for _ in range(9)]
k_text = ax.text(0.05,0.85,'',transform=ax.transAxes)
textTemplate = 't = %.3f days\n'
# v = np.random.rand(nn) * 100 * 1000
# x = r * np.cos(theta).astype(np.double)
# y = r * np.sin(theta).astype(np.double)
# v_x = -v * np.sin(theta).astype(np.double)
# v_y = v * np.cos(theta).astype(np.double)

ts = np.arange(0, N * dt, dt)
xxs, yys = [], []
accx_0 = np.zeros(len(m))
accy_0 = np.zeros(len(m))
accx_1 = np.zeros(len(m))
accy_1 = np.zeros(len(m))
x_ij = (x - x.reshape(len(m), 1))
y_ij = (y - y.reshape(len(m), 1))
# print(x_ij[0])
r_ij = np.sqrt(x_ij ** 2 + y_ij ** 2)
for i in range(len(m)):
    for j in range(len(m)):
        if i != j:
            accx_0[i] += (m[j] * x_ij[i, j] / r_ij[i, j] ** 3)
            accy_0[i] += (m[j] * y_ij[i, j] / r_ij[i, j] ** 3)
    # print(accx_0[i], accy_0[i])
x += v_x * dt + 0.5 * accx_0 * dt ** 2
y += v_y * dt + 0.5 * accy_0 * dt ** 2
xxs.append(x.tolist())
yys.append(y.tolist())

# n = 10
# a_np = np.random.randn(n).astype(np.double)
# b_np = np.random.randn(n).astype(np.double)
# print(a_np,b_np)
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
# a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
# b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
m_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m)
x_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)
y_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y)
v_x_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=v_x)
v_y_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=v_y)
accx_0_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=accx_0)
accy_0_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=accy_0)
accx_1_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=accx_1)
accy_1_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=accy_1)

prg = cl.Program(ctx, '''
    int N=500;
    int dt=360000;
    int nn=9;
    __kernel void openclverlet(
      __global double *m_g,__global double *accx_0_g,__global double *accy_0_g,__global double *accx_1_g,__global double *accy_1_g,__global double *x_g,__global double *y_g,__global double *v_x_g,__global double *v_y_g,__global double *xs,__global double *ys)
    {
      int gid = get_global_id(0);      
      /*double x_ij[nn][nn]={};
      double y_ij[nn][nn]={};
      double r_ij[nn][nn]={};
      for(int i=0;i<nn;i++){
         for(int j=0;j<nn;j++){
            x_ij[i][j]=x_g[j]-x_g[i];
            y_ij[i][j]=y_g[j]-y_g[i];
            r_ij[i][j]=sqrt(x_ij[i][j]*x_ij[i][j]+y_ij[i][j]*y_ij[i][j]);*/
      for(int j=0;j<nn;j++){
        if(gid!=j){
          accx_1_g[gid]+=(m_g[j] * (x_g[j]-x_g[gid]) / pow(sqrt(pow(x_g[j]-x_g[gid],2)+pow(y_g[j]-y_g[gid],2)),3));
          accy_1_g[gid]+=(m_g[j] * (y_g[j]-y_g[gid]) / pow(sqrt(pow(x_g[j]-x_g[gid],2)+pow(y_g[j]-y_g[gid],2)),3));

        }
      }      




      //printf("m_g=%d ",nn);
      //double x_ij[m_g.size]
      //printf("%d ",gid);
      //xs[gid]=x_g[gid]+y_g[gid]+N;
      //ys[gid]=2*(x_g[gid]+y_g[gid]);


    }
''').build()

start = time.time()

xs = cl.Buffer(ctx, mf.READ_WRITE, x.nbytes)
ys = cl.Buffer(ctx, mf.READ_WRITE, y.nbytes)
for _ in ts:
    knl = prg.openclverlet
    # print(x.shape,y.shape,v_x.shape,v_y.shape)
    knl(queue, (nn,), None, m_g, accx_0_g, accy_0_g, accx_1_g, accy_1_g, x_g, y_g, v_x_g, v_y_g, xs, ys)

    res_ax_1 = np.empty_like(x)
    cl.enqueue_copy(queue, res_ax_1, accx_1_g)
    res_ay_1 = np.empty_like(y)
    cl.enqueue_copy(queue, res_ay_1, accy_1_g)

    res_ax_0 = np.empty_like(x)
    cl.enqueue_copy(queue, res_ax_0, accx_0_g)
    res_ay_0 = np.empty_like(y)
    cl.enqueue_copy(queue, res_ay_0, accy_0_g)

    # print(np.array(res_ax_0))
    res_v_x_g = np.empty_like(x)
    cl.enqueue_copy(queue, res_v_x_g, v_x_g)
    res_v_y_g = np.empty_like(y)
    cl.enqueue_copy(queue, res_v_y_g, v_y_g)
    res_x_g = np.empty_like(x)
    cl.enqueue_copy(queue, res_x_g, x_g)
    res_y_g = np.empty_like(y)
    cl.enqueue_copy(queue, res_y_g, y_g)

    res_v_x_g = np.array(res_v_x_g) + 0.5 * (np.array(res_ax_0) + np.array(res_ax_1)) * dt
    res_v_y_g += 0.5 * (np.array(res_ay_0) + np.array(res_ay_1)) * dt
    res_x_g += res_v_x_g * dt + 0.5 * np.array(res_ax_1) * dt ** 2
    res_y_g += res_v_y_g * dt + 0.5 * np.array(res_ay_1) * dt ** 2

    xxs.append(res_x_g.tolist())
    yys.append(res_y_g.tolist())

    accx_0_g = accx_1_g
    accy_0_g = accy_1_g
    accx_1 = np.zeros(nn)
    accy_1 = np.zeros(nn)
    accx_1_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=accx_1)
    accy_1_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=accy_1)

    x_g=cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=res_x_g)
    y_g=cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=res_y_g)
    v_x_g=cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=res_v_x_g)
    v_y_g=cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=res_v_y_g)


print(time.time() - start)
xs = np.array(xxs)
ys = np.array(yys)
xxs = np.array(xxs)
yys = np.array(yys)
dataframe = pd.DataFrame(
        {'x1': xs[:, 0], 'y1': ys[:, 0], 'x2': xs[:, 1], 'y2': ys[:, 1], 'x3': xs[:, 2], 'y3': ys[:, 2],
         'x4': xs[:, 3], 'y4': ys[:, 3], 'x5': xs[:, 4], 'y5': ys[:, 4], 'x6': xs[:, 5], 'y6': ys[:, 5],
         'x7': xs[:, 6], 'y7': ys[:, 6], 'x8': xs[:, 7], 'y8': ys[:, 7], 'x9': xs[:, 8], 'y9': ys[:, 8]})
dataframe.to_csv("openclsolar.csv", index=False, sep=',')
# print(res_x)
# print(res_y)


def animate(n):
    print(n)
    for i in range(len(m)):
        traces[i].set_data(xxs[:n,i],yys[:n,i])
        pts[i].set_data(xxs[n,i],yys[n,i])
    #k_text.set_text(textTemplate % (ts[n]/3600/24))
    return traces+pts+[k_text]

ani = FuncAnimation(fig, animate,
    range(N), interval=100, blit=True)
plt.show()
