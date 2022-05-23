import matplotlib.pyplot as plt
import matplotlib.tri as tri
from dolfin import *
from mshr.cpp import Circle, generate_mesh
from mshr import *
from fenics import *
import numpy as np

def boundary_L(x, on_boundary):
    return on_boundary and x[0]<0

def visualization(mesh,u,title):
    n = mesh.num_vertices()
    d = mesh.geometry().dim()
    mesh_coordinates = mesh.coordinates().reshape((n, d))
    triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
    triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)
    plt.figure()
    zfaces = (np.asarray([u(cell.midpoint()) for cell in cells(mesh)]))
    plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    plt.grid()
    plt.colorbar()
    plt.title(title)
    plt.savefig(title +'.jpg')
    plt.plot()
    plt.show()

r = Expression("sqrt(x[0] * x[0] + x[1] * x[1])", degree = 2)
phi = Expression("atan2(x[1], x[0])", degree = 2)

#test1
#u=1+r * r+r* sin(phi), R=1, alpha=1
#error_L2  = 0.6161284414494278
#error_max = 0.5891074781400436
'''
R=1
alpha=Constant(1)

domain = Circle(Point(0 , 0 ) , R )
mesh = generate_mesh(domain , 64 )
V = FunctionSpace(mesh, "P", 2)
u_D = Expression ('1+r * r+r* sin(phi)',r=r,phi=phi, degree=2)

f = Expression("-6+alpha * r * (1+r * r+r* sin(phi))", alpha = alpha, r = r, phi = phi, degree = 2)
h = Expression("2 + sin(phi)", phi = phi, degree = 2)
g = Expression("1+2+sin(phi)", phi = phi, degree = 2)
bc = DirichletBC(V, h, boundary_L)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = (dot(grad(u), grad(v)) + alpha * u * v) * dx
L = f*v*dx+g*v*ds

u=Function(V)
solve( a == L , u,bc)

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)

error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)

visualization(mesh,u,"1_Finite element solution")
visualization(mesh,u_D,"1_Exact solution")
'''

#test2
#u=r*cos(phi),R=1, alpha=1
#error_L2  = 5.4772825847509644e-05
#error_max = 0.00012214305934699077
'''
R=1
alpha=Constant(1)

domain = Circle(Point(0 , 0 ) , R )
mesh = generate_mesh(domain , 64 )
V = FunctionSpace(mesh, "P", 2)
u_D = Expression ('r * cos(phi)',r=r,phi=phi, degree=2)


f = Expression("alpha * r * cos(phi)", alpha = alpha, r = r, phi = phi, degree = 2)
h = Expression("1 * cos(phi)", phi = phi, degree = 2)
g = Expression("cos(phi)", phi = phi, degree = 2)
bc = DirichletBC(V, h, boundary_L)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = (dot(grad(u), grad(v)) + alpha * u * v) * dx
L = f*v*dx+g*v*ds

u=Function(V)
solve( a == L , u,bc)

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)

error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)

visualization(mesh,u,"2_Finite element solution")
visualization(mesh,u_D,"2_Exact solution")
'''

#test3
# u = r * sin(phi), R = 1, alpha = 1
#error_L2  = 6.011363604090357e-05
#error_max = 0.00012213932955829065

R=1
alpha=Constant(1)

domain = Circle(Point(0 , 0 ) , R )
mesh = generate_mesh(domain , 64 )
V = FunctionSpace(mesh, "P", 2)
u_D = Expression ('r * sin(phi)',r=r,phi=phi, degree=2)


f = Expression("alpha * r * sin(phi)", alpha = alpha, r = r, phi = phi, degree = 2)
h = Expression("1 * sin(phi)", phi = phi, degree = 2)
g = Expression("sin(phi)", phi = phi, degree = 2)
bc = DirichletBC(V, h, boundary_L)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = (dot(grad(u), grad(v)) + alpha * u * v) * dx
L = f*v*dx+g*v*ds

u=Function(V)
solve( a == L , u,bc)

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)

error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)

visualization(mesh,u,"3_Finite element solution")
visualization(mesh,u_D,"3_Exact solution")


