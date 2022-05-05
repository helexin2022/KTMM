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
        long double au= 1.48e11
        long double G=6.67e-11
        long double RE=1.48e11
        long double ME=5.965e24
        np.ndarray[long double, ndim=1] m= np.array([3.32e5, 0.055, 0.815, 1.0,
                  0.107, 317.8, 95.16, 14.54, 17.14]) * ME * 6.67e-11
        np.ndarray[long double, ndim=1] r = np.array([0.0, 0.387, 0.723, 1.0, 1.524, 5.203,
                  9.537, 19.19, 30.7]) * RE
        np.ndarray[long double, ndim=1] theta = np.array([0.90579977, 4.76568695, 1.34869972, 6.02969388, 2.24714959, 3.45095948,
             3.41281759, 4.32174632, 2.33019222])
        np.ndarray[long double, ndim=1] x =  r * np.cos(theta)
        np.ndarray[long double, ndim=1] y =  r * np.sin(theta)
        np.ndarray[long double, ndim=1] v = np.array([0, 47.89, 35.03, 29.79, 24.13, 13.06, 9.64, 6.81, 5.43]) * 1000
        np.ndarray[long double, ndim=1] v_x =  -v * np.sin(theta)
        np.ndarray[long double, ndim=1] v_y =  v * np.cos(theta)



    # m = np.random.rand(nn) * 1000 * ME * 6.67e-11
    # r = np.random.rand(nn) * 10 * RE
    # theta = np.random.rand(nn) * np.pi * 2
    # v = np.random.rand(nn) * 100 * 1000
    # x = r * np.cos(theta)
    # y = r * np.sin(theta)
    # v_x = -v * np.sin(theta)
    # v_y = v * np.cos(theta)
    #v = [0.00, 47.89, 35.03, 29.79, 24.13, 13.06, 9.64, 6.81, 5.43] * 1000
    #v_x =  -v * np.sin(theta)
    #v_y =  v * np.cos(theta)
    return m,x,y,v_x,v_y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def pythonverlet(int N,int dt,np.ndarray[long double, ndim=1] m,np.ndarray[long double, ndim=1] x,np.ndarray[long double, ndim=1] y,np.ndarray[long double, ndim=1] v_x,np.ndarray[long double, ndim=1] v_y):
    cdef:
        np.ndarray[long, ndim=1] ts = np.arange(0, N * dt, dt)
        np.ndarray[long double, ndim=1] accx_0 = np.zeros(len(m))
        np.ndarray[long double, ndim=1] accy_0 = np.zeros(len(m))
        np.ndarray[long double, ndim=1] accx_1 = np.zeros(len(m))
        np.ndarray[long double, ndim=1] accy_1 = np.zeros(len(m))

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

@cython.boundscheck(False)
@cython.wraparound(False)
def main(nn):
    cdef:
        long double au= 1.48e11
        long double G=6.67e-11
        long double RE=1.48e11
        long double ME=5.965e24
        long double start=time.time()
        int N = 500
        int dt = 360000
    m, x, y, v_x, v_y = create(nn)
    xs, ys = pythonverlet(N, dt, m, x, y, v_x, v_y)
    cdef double end = time.time()

    dataframe = pd.DataFrame(
        {'x1': xs[:, 0], 'y1': ys[:, 0], 'x2': xs[:, 1], 'y2': ys[:, 1], 'x3': xs[:, 2], 'y3': ys[:, 2],
         'x4': xs[:, 3], 'y4': ys[:, 3], 'x5': xs[:, 4], 'y5': ys[:, 4], 'x6': xs[:, 5], 'y6': ys[:, 5],
         'x7': xs[:, 6], 'y7': ys[:, 6], 'x8': xs[:, 7], 'y8': ys[:, 7], 'x9': xs[:, 8], 'y9': ys[:, 8]})
    dataframe.to_csv("cysolar.csv", index=False, sep=',')

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





