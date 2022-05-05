import time
import numpy as np
import random as random
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

au, G, RE, ME = 1.48e11, 6.67e-11, 1.48e11, 5.965e24
def particlesGeneration(nn):

    # m = np.array([3.32e5,0.055,0.815,1,
    #               0.107,317.8,95.16,14.54,17.14])*ME*6.67e-11
    #
    # r = np.array([0,0.387,0.723,1,1.524,5.203,
    #               9.537,19.19,30.7])*RE
    # m=np.random.rand(nn)*1000*ME*6.67e-11
    # r=np.random.rand(nn)*10*RE
    # theta = np.random.rand(nn)*np.pi*2
    # v=np.random.rand(nn)*100*1000
    m=np.loadtxt('50m.txt')
    r=np.loadtxt('50r.txt')
    theta=np.loadtxt('50theta.txt')
    v=np.loadtxt('50v.txt')
    # m = np.loadtxt('100m.txt')
    # r = np.loadtxt('100r.txt')
    # theta = np.loadtxt('100theta.txt')
    # v = np.loadtxt('100v.txt')
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    v_x = -v * np.sin(theta)
    v_y = v * np.cos(theta)
    return np.array(m), np.array(x), np.array(y), np.array(v_x),np.array(v_y)
# Multiprocessing usage

def pythonverlet(m,x,y,v_x,v_y,dt,N):
    ts = np.arange(0, N * dt, dt)
    xs, ys = [], []
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
    xs.append(x.tolist())
    ys.append(y.tolist())

    for _ in ts:
        x_ij = (x - x.reshape(len(m), 1))
        y_ij = (y - y.reshape(len(m), 1))
        r_ij = np.sqrt(x_ij ** 2 + y_ij ** 2)
        for i in range(len(m)):
            for j in range(len(m)):
                if i != j:
                    accx_1[i] += (m[j] * x_ij[i, j] / r_ij[i, j] ** 3)
                    accy_1[i] += (m[j] * y_ij[i, j] / r_ij[i, j] ** 3)
                    # print(accx_1[i], accy_1[i])
        v_x += 0.5 * (accx_0 + accx_1) * dt
        v_y += 0.5 * (accy_0 + accy_1) * dt
        x += v_x * dt + 0.5 * accx_1 * dt ** 2
        y += v_y * dt + 0.5 * accy_1 * dt ** 2
        # print(x,y)
        xs.append(x.tolist())
        ys.append(y.tolist())
        accx_0 = accx_1
        accy_0 = accy_1
        accx_1 = np.zeros(len(m))
        accy_1 = np.zeros(len(m))
    xs = np.array(xs)
    ys = np.array(ys)
    #print(x_ij)
    return xs,ys


if __name__ == '__main__':
    nn=50
    m,x,y,v_x,v_y=particlesGeneration(nn)
    #print(mp.cpu_count())
    start = time.time()
    pool = mp.Pool(5)
    N=500
    dt=360000
    #p = pool.starmap(pythonverlet, [(m,x,y,v_x,v_y,dt,N)])
    result= pool.apply_async(pythonverlet, (m,x,y,v_x,v_y,dt,N,))

    print(time.time() - start)
    #print(len(result.get()[0]))
    # print(len(p[0][0]))
    # print(p[0][0])
    pool.close()
    pool.join()
    print(time.time() - start)
    xs = result.get()[0]
    ys = result.get()[1]

    # print(len(p[0][0]))
    # print(p[0][0])
    def animate(n):
        for i in range(len(m)):
            traces[i].set_data(xs[:n, i], ys[:n, i])
            pts[i].set_data(xs[n, i], ys[n, i])
        # k_text.set_text(textTemplate % (ts[n]/3600/24))
        return traces + pts + [k_text]


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(xlim=(-31 * RE, 31 * RE), ylim=(-31 * RE, 31 * RE))
    ax.grid()
    traces = [ax.plot([], [], '-', lw=0.5)[0] for _ in range(nn)]
    pts = [ax.plot([], [], marker='o')[0] for _ in range(nn)]
    k_text = ax.text(0.05, 0.85, '', transform=ax.transAxes)
    # textTemplate = 't = %.3f days\n'
    ani = FuncAnimation(fig, animate,
                        range(N), interval=100, blit=True)
    plt.show()
