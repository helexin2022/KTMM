from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from mshr.cpp import Circle, generate_mesh
from mshr import *
import matplotlib.tri as tri
import matplotlib as mpl
import matplotlib
from PIL import Image
#import imageio
#from images2gif import writeGif
#import cv2
# import skbuild


def boundary( x , on_boundary ) :
    return on_boundary and x[0]<0

def maxmin(mesh,u_list):
    min=100
    max=-100
    for i,u in enumerate(u_list):
        n = mesh.num_vertices()
        d = mesh.geometry().dim()
        mesh_coordinates = mesh.coordinates().reshape((n, d))
        zfaces = (np.asarray([u(cell.midpoint()) for cell in cells(mesh)]))

        minn=np.amin(zfaces)
        if(min>minn):
            min=minn

        maxx=np.amax(zfaces)
        if(maxx>max):
            max=maxx
    return max,min


def visualization(mesh,u,i,max,min,title):
    n = mesh.num_vertices()
    d = mesh.geometry().dim()
    mesh_coordinates = mesh.coordinates().reshape((n, d))
    triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
    triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)
    fig=plt.figure()
    plt.axis('equal')
    zfaces = (np.asarray([u(cell.midpoint()) for cell in cells(mesh)]))
    cmp = mpl.colors.LinearSegmentedColormap.from_list("",['r', 'g', 'b', 'y'])
    plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k',cmap = cmp)
    plt.clim(min, max)
    plt.grid()
    bounds = np.linspace(min, max, 5)
    plt.title(title)
    ax = fig.add_axes([0.91, 0.12, 0.02, 0.757])  # 第一个左右移动，第二个上下移动，第三个左右缩放，第四个上下缩放
    #ax = fig.add_axes([0.9, 0.15, 0.05, 0.76])
    matplotlib.colorbar.ColorbarBase(ax, orientation='vertical', cmap=cmp, ticks=bounds, boundaries=bounds,
                                          norm=matplotlib.colors.Normalize(vmin=min, vmax=max))
    #plt.colorbar()
    plt.plot()
    plt.savefig(f"./graphs_test3/u_{i}_{title}.jpg")
    #plt.savefig(f"./graphs_test onclass/u_{i}_{title}.jpg")
    plt.show()
    plt.close()

r = Expression("sqrt(x[0] * x[0] + x[1] * x[1])", degree = 2)
phi = Expression("atan2(x[1], x[0])", degree = 2)

T =20.0
num_steps = 20
dt = T / num_steps

#test1 u=1+x^2+3y^2+1.2t=1+r^2+2r*r*sin(phi)*sin(phi)+1.2t, alpha=1,R=1
'''
R=1
domain = Circle(Point(0 , 0 ) , R )
mesh = generate_mesh(domain , 32 )
V = FunctionSpace(mesh,'P',2)

t=0
u_D = Expression("1+r*r+2*r*r*sin(phi)*sin(phi)+1.2*t", t = t, r = r, phi = phi, degree = 2)
alpha = Constant(1)
f = Constant(-6.8)#Expression("-6.8", t = t, r = r, phi = phi, degree = 2)
h = Expression("2+2*sin(phi)*sin(phi)+1.2*t", t = t, phi = phi, degree = 2)
g = Expression("2+4*sin(phi)*sin(phi)", r=r,t = t, phi = phi, degree = 2)


u_n = interpolate(h, V)
# F = (1/(dt*alpha))*u*v*dx + dt * dot ( grad ( u ) , grad ( v ) ) *dx - (u_n + dt * f ) *v*dx
# a , L = lhs(F) , rhs(F)
# a = (dot(grad(u), grad(v)) + (1/(dt*alpha)) * u * v) * dx
# L = (u_n/(dt*alpha)+f/alpha) * v * dx + g * v * ds
# u=Function(V)
t =0

u_list_fes = [u_n.copy()]
u_list_exact = [Expression("1+r*r+2*r*r*sin(phi)*sin(phi)+1.2*t", t = t, r = r, phi = phi, degree = 2)]

error_L2_List=[]
error_max_List=[]
for n in range(num_steps):
    # update current time
    t += dt
    u_D.t = t
    f.t = t
    h.t = t
    g.t = t
    u = TrialFunction(V)
    v = TestFunction(V)
    bc = DirichletBC(V, u_D, boundary)
    a = (dot(grad(u), grad(v)) + (1 / (dt * alpha)) * u * v) * dx
    L = (u_n / (dt * alpha) + f / alpha) * v * dx + g * v * ds
    u = Function(V)
    solve(a==L,u,bc)
    # plot(u)
    # plt.show()
    # visualization(mesh,u,n,'Finite element solution')
    #error = np.abs(np.array(u_e.vector())-np.array(u.vector())).max()
    #print('t={}:error={}'.format(t,error))
    # Compute error in L2 norm
    error_L2 = errornorm(u_D, u, 'L2')

    # Compute maximum error at vertices
    vertex_values_u_D = u_D.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)

    error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))
    error_L2_List.append(error_L2)
    error_max_List.append(error_max)
    # Print errors
    print("t=",t)
    print('error_L2  =', error_L2)
    print('error_max =', error_max)
    print("-"*50)
    #update previous solution
    u_n.assign(u)
    u_list_fes.append(u_n.copy())
    u_list_exact.append(Expression("1+r*r+2*r*r*sin(phi)*sin(phi)+1.2*t", t=t, r=r, phi=phi, degree=2))  # deep copy doesn't work!
p=np.linspace(dt,T,num_steps)
pp=np.array(error_L2_List)
ppp=np.array(error_max_List)
# print(pp)
# print(len(p),len(error_L2_List))
np.savetxt('error_List_num.txt',p)
np.savetxt('error_L2_List.txt',pp)
np.savetxt('error_max_List.txt',ppp)

maxfes,minfes=maxmin(mesh,u_list_fes)
print("max,min=",maxfes,minfes)
for n in range(num_steps):
    visualization(mesh, u_list_fes[n], n,maxfes,minfes, 'Finite element solution')

maxes,mines=maxmin(mesh,u_list_exact)
print("max,min=",maxes,mines)
for n in range(num_steps):
    visualization(mesh, u_list_exact[n], n,maxfes,minfes, 'Exact solution')
'''

#test2 u=1+x^2+4y^2+t=1+r^2+3r^2 sin(phi)+t,alpha=1,R=1
'''
R=1
domain = Circle(Point(0 , 0 ) , R )
mesh = generate_mesh(domain , 32 )
V = FunctionSpace(mesh,'P',2)

t=0
u_D = Expression("1+r*r+3*r*r*sin(phi)*sin(phi)+t", t = t, r = r, phi = phi, degree = 2)
alpha = Constant(1)
f = Constant(-9)
h = Expression("2+3*sin(phi)*sin(phi)+t", t = t, phi = phi, degree = 1)
g = Expression("2+6*sin(phi)*sin(phi)", t = t, R=R,phi = phi, degree = 1)


u_n = interpolate(u_D, V)
# F = (1/(dt*alpha))*u*v*dx + dt * dot ( grad ( u ) , grad ( v ) ) *dx - (u_n + dt * f ) *v*dx
# a , L = lhs(F) , rhs(F)
# a = (dot(grad(u), grad(v)) + (1/(dt*alpha)) * u * v) * dx
# L = (u_n/(dt*alpha)+f/alpha) * v * dx + g * v * ds
# u=Function(V)
t =0

u_list_fes = [u_n.copy()]
u_list_exact = [Expression("1+r*r+3*r*r*sin(phi)*sin(phi)+t", t = t, r = r, phi = phi, degree = 2)]
error_L2_List=[]
error_max_List=[]
for n in range(num_steps):
    # update current time
    t += dt
    u_D.t = t
    f.t = t
    h.t = t
    g.t = t
    u = TrialFunction(V)
    v = TestFunction(V)
    bc = DirichletBC(V, u_D, boundary)
    a = (dot(grad(u), grad(v)) + (1/(dt * alpha)) * u * v) * dx
    L = (u_n / (dt*alpha) + f / alpha) * v * dx + g * v * ds

    u = Function(V)
    solve(a==L,u,bc)
    # plot(u)
    # plt.show()
    # visualization(mesh,u,n,'Finite element solution')
    #error = np.abs(np.array(u_e.vector())-np.array(u.vector())).max()
    #print('t={}:error={}'.format(t,error))
    # Compute error in L2 norm
    error_L2 = errornorm(u_D, u, 'L2')

    # Compute maximum error at vertices
    vertex_values_u_D = u_D.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)

    error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))
    error_L2_List.append(error_L2)
    error_max_List.append(error_max)
    # Print errors
    print("t=",t)
    print('error_L2  =', error_L2)
    print('error_max =', error_max)
    print("-"*50)
    #update previous solution
    u_n.assign(u)
    u_list_fes.append(u_n.copy())
    u_list_exact.append(Expression("1+r*r+3*r*r*sin(phi)*sin(phi)+t", t=t, r=r, phi=phi, degree=2))  # deep copy doesn't work!

p=np.linspace(dt,T,num_steps)
pp=np.array(error_L2_List)
ppp=np.array(error_max_List)
# print(pp)
# print(len(p),len(error_L2_List))
np.savetxt('error_List_num2.txt',p)
np.savetxt('error_L2_List2.txt',pp)
np.savetxt('error_max_List2.txt',ppp)

maxfes,minfes=maxmin(mesh,u_list_fes)
print("max,min=",maxfes,minfes)
for n in range(num_steps):
    visualization(mesh, u_list_fes[n], n,maxfes,minfes, 'Finite element solution')

maxes,mines=maxmin(mesh,u_list_exact)
print("max,min=",maxes,mines)
for n in range(num_steps):
    visualization(mesh, u_list_exact[n], n,maxes,mines, 'Exact solution')
'''

#test3 u=9+2*x^2+3*y^2+4t=9+2*r^2+r^2*sin(phi)^2+4t,alpha=1,R=1

R=1
domain = Circle(Point(0 , 0 ) , R )
mesh = generate_mesh(domain , 32 )
V = FunctionSpace(mesh,'P',2)

t=0
u_D = Expression("9+2*r*r+r*r*sin(phi)*sin(phi)+4*t", t = t, r = r, phi = phi, degree = 2)
alpha = Constant(1)
f = Constant(-6)
h = Expression("11+sin(phi)*sin(phi)+4*t", t = t, phi = phi, degree = 1)
g = Expression("4+2*sin(phi)*sin(phi)", t = t, R=R,phi = phi, degree = 1)


u_n = interpolate(u_D, V)
# F = (1/(dt*alpha))*u*v*dx + dt * dot ( grad ( u ) , grad ( v ) ) *dx - (u_n + dt * f ) *v*dx
# a , L = lhs(F) , rhs(F)
# a = (dot(grad(u), grad(v)) + (1/(dt*alpha)) * u * v) * dx
# L = (u_n/(dt*alpha)+f/alpha) * v * dx + g * v * ds
# u=Function(V)
t =0

u_list_fes = [u_n.copy()]
u_list_exact = [Expression("9+2*r*r+r*r*sin(phi)*sin(phi)+4*t", t = t, r = r, phi = phi, degree = 2)]
error_L2_List=[]
error_max_List=[]
for n in range(num_steps):
    # update current time
    t += dt
    u_D.t = t
    f.t = t
    h.t = t
    g.t = t
    u = TrialFunction(V)
    v = TestFunction(V)
    bc = DirichletBC(V, u_D, boundary)
    a = (dot(grad(u), grad(v)) + (1/(dt * alpha)) * u * v) * dx
    L = (u_n / (dt*alpha) + f / alpha) * v * dx + g * v * ds

    u = Function(V)
    solve(a==L,u,bc)
    # plot(u)
    # plt.show()
    # visualization(mesh,u,n,'Finite element solution')
    #error = np.abs(np.array(u_e.vector())-np.array(u.vector())).max()
    #print('t={}:error={}'.format(t,error))
    # Compute error in L2 norm
    error_L2 = errornorm(u_D, u, 'L2')

    # Compute maximum error at vertices
    vertex_values_u_D = u_D.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)

    error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))
    error_L2_List.append(error_L2)
    error_max_List.append(error_max)
    # Print errors
    print("t=",t)
    print('error_L2  =', error_L2)
    print('error_max =', error_max)
    print("-"*50)
    #update previous solution
    u_n.assign(u)
    u_list_fes.append(u_n.copy())
    u_list_exact.append(Expression("9+2*r*r+r*r*sin(phi)*sin(phi)+4*t", t=t, r=r, phi=phi, degree=2))  # deep copy doesn't work!

p=np.linspace(dt,T,num_steps)
pp=np.array(error_L2_List)
ppp=np.array(error_max_List)
# print(pp)
# print(len(p),len(error_L2_List))
np.savetxt('error_List_num3.txt',p)
np.savetxt('error_L2_List3.txt',pp)
np.savetxt('error_max_List3.txt',ppp)

maxfes,minfes=maxmin(mesh,u_list_fes)
print("max,min=",maxfes,minfes)
for n in range(num_steps):
    visualization(mesh, u_list_fes[n], n,maxfes,minfes, 'Finite element solution')

maxes,mines=maxmin(mesh,u_list_exact)
print("max,min=",maxes,mines)
for n in range(num_steps):
    visualization(mesh, u_list_exact[n], n,maxes,mines, 'Exact solution')


#test onclass

