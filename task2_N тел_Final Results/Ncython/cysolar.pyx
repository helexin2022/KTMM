cimport cython
cimport numpy as np
import numpy as np
import time
from libc.math cimport sqrt, pow,sin,cos
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
import pandas as pd

@cython.boundscheck(False)
@cython.wraparound(False)
def create (int nn):
    cdef:
        double au= 1.48e11
        double G=6.67e-11
        double RE=1.48e11
        double ME=5.965e24
        #np.ndarray[double, ndim=1] m= np.random.rand(nn) * 1000 * ME * 6.67e-11
        #np.ndarray[double, ndim=1] r = np.random.rand(nn) * 10 * RE
        #np.ndarray[double, ndim=1] theta = np.random.rand(nn) * np.pi * 2
        #np.ndarray[double, ndim=1] v = np.random.rand(nn) * 100 * 1000

        np.ndarray[double, ndim=1] m=np.loadtxt('50m.txt')
        np.ndarray[double, ndim=1] r=np.loadtxt('50r.txt')
        np.ndarray[double, ndim=1] theta=np.loadtxt('50theta.txt')
        np.ndarray[double, ndim=1] v=np.loadtxt('50v.txt')

        np.ndarray[double, ndim=1] x =  r * np.cos(theta)
        np.ndarray[double, ndim=1] y =  r * np.sin(theta)
        np.ndarray[double, ndim=1] v_x =  -v * np.sin(theta)
        np.ndarray[double, ndim=1] v_y =  v * np.cos(theta)



    # m = np.random.rand(nn) * 1000 * ME * 6.67e-11
    # r = np.random.rand(nn) * 10 * RE
    # theta = np.random.rand(nn) * np.pi * 2
    # v = np.random.rand(nn) * 100 * 1000
    # x = r * np.cos(theta)
    # y = r * np.sin(theta)
    # v_x = -v * np.sin(theta)
    # v_y = v * np.cos(theta)
    return m,x,y,v_x,v_y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def pythonverlet(int N,int dt,np.ndarray[double, ndim=1] m,np.ndarray[double, ndim=1] x,np.ndarray[double, ndim=1] y,np.ndarray[double, ndim=1] v_x,np.ndarray[double, ndim=1] v_y):
    cdef:
        np.ndarray[long, ndim=1] ts = np.arange(0, N * dt, dt)
        np.ndarray[double, ndim=1] accx_0 = np.zeros(len(m))
        np.ndarray[double, ndim=1] accy_0 = np.zeros(len(m))
        np.ndarray[double, ndim=1] accx_1 = np.zeros(len(m))
        np.ndarray[double, ndim=1] accy_1 = np.zeros(len(m))

    xs, ys = [], []
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


    return xs,ys

def main(nn):
    cdef:
        double au= 1.48e11
        double G=6.67e-11
        double RE=1.48e11
        double ME=5.965e24
        double start=time.time()
        int N = 500
        int dt = 360000
    m, x, y, v_x, v_y = create(nn)
    xs, ys = pythonverlet(N, dt, m, x, y, v_x, v_y)
    cdef double end = time.time()


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
    #textTemplate = 't = %.3f days\n'
    ani = FuncAnimation(fig, animate,
                        range(N), interval=100, blit=True)
    plt.show()

    return (end-start)





