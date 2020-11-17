#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:55:21 2020

@author: jack
"""

# Imports 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import integrate
import time
import math
plt.style.use("classic")

"""
To Do:
    Fix error vector for pressure plot
    Make log-log loop
    

"""
#%###########################

# Objects


class Freestream():
    
    def __init__(self, u_inf, alpha=0):
        
        self.u_inf = u_inf
        self.alpha = alpha
    
    def GetStreamFunction(self, Xmesh, Ymesh):
        psi_mesh = self.u_inf * Ymesh * np.cos(self.alpha) + self.u_inf*Xmesh*np.sin(self.alpha)
    
        return psi_mesh
    
    def GetStreamPotential(self, Xmesh, Ymesh):
        
        phi_mesh = self.u_inf * Ymesh * np.sin(self.alpha) + self.u_inf*Xmesh*np.cos(self.alpha)

        return phi_mesh

    def GetVelocity(self, Xmesh, Ymesh):
        """
        Returns stream velocity mesh grids for the source.

        Parameters
        ----------
        Xmesh : Control X meshgrid
        Ymesh : Control Y meshgrid

        Returns
        -------
        umesh : u component meshgrid
        vmesh : v component meshgrid

        """
        umesh = Xmesh * 0 + self.u_inf*np.cos(self.alpha)
        vmesh = Xmesh * 0 + self.u_inf*np.sin(self.alpha)
         
        return umesh, vmesh

    def GetCp(self, Xmesh, Ymesh):
        Cp = Xmesh*0
        
        return Cp


class Panel():
    
    def __init__(self, x1, y1, x2, y2):
        
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        
        # Take the center points as the average of the two points
        
        self.xc = 0.5 * (x1 + x2)
        self.yc = 0.5 * (y1 + y2)
        
        self.length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        self.strength = 0
        self.v_tan = 0
        self.surface_cp = 0
        
        
        # if x2 -x1 <= 0:
        #     self.beta = np.arccos((y2 - y1) / self.length)
        # elif x2 - x1 >= 0:
        #     self.beta = np.pi + np.arccos(-(y2 - y1) / self.length)
        
        if x2 - x1 <= 0.0:
            self.beta = math.acos((y2 - y1) / self.length)
        elif x2 - x1 > 0.0:
            self.beta = math.pi + math.acos(-(y2 - y1) / self.length)
        
        if self.beta <= math.pi:
            self.loc = 'upper'
        else:
            self.loc = 'lower'

    def GetVelocity(self, X, Y):
        vec_intregral = np.vectorize(self.Integral)
        u = self.strength / (2 * np.pi) * vec_intregral(X, Y, 1, 0)
        v = self.strength / (2 * np.pi) * vec_intregral(X, Y, 0, 1)
        
        return u, v
    
    def GetStreamFunction(self, Xmesh, Ymesh):
        """
        Finds meshgrid for streamfuntion psi

        Parameters
        ----------
        Xmesh : Control X meshgrid
        Ymesh : Control Y meshgrid

        Returns
        -------
        psi_mesh : Stream function meshgrid

        """
        psi_mesh = Xmesh*0
        return psi_mesh
    
    def GetStreamPotential(self, Xmesh, Ymesh):
        
        phi_mesh = Xmesh*0

        return phi_mesh

#%###########################

# Functions

def ConstructCylinder(xc, yc, R, N):
    
    xpoints = R * np.cos(np.linspace(0, 2*np.pi, N+1))
    ypoints = R * np.sin(np.linspace(0, 2*np.pi, N+1))
    
    panel_list = np.empty((N), dtype=object)
    for i in range(N):
        panel_list[i] = Panel(xpoints[i], ypoints[i], xpoints[i+1], ypoints[i+1])
    return xpoints, ypoints, panel_list


def PanelStrengthSolver(freestream, panel_list):
    N = len(panel_list)
    A = np.zeros((N, N))
    
    # For every panel i
    for i, panel_i in enumerate(panel_list):
        
        # For every other panel j
        for j, panel_j in enumerate(panel_list):
            
            # Can't induce flow on itself
            if i == j:
                A[i, j] = 0.5
                
            else:
                # Flow induced on panel i by panel j
                 A[i, j] = 1 / (2*np.pi) * I(panel_i.xc,
                                             panel_i.yc,
                                             panel_j,
                                             np.cos(panel_i.beta),
                                             np.sin(panel_i.beta))
            
    solution_vector = np.zeros((N, 1))
    for i in range(N):
        solution_vector[i] = -freestream.u_inf * np.cos(freestream.alpha - panel_list[i].beta)
    
    strength_list = np.linalg.solve(A, solution_vector)
    for i, panel in enumerate(panel_list):
        panel.strength = strength_list[i]
    return panel_list

def PanelTangentSolver(freestream, panel_list):
    N = len(panel_list)
    A = np.zeros((N,N))
    
    # for every panel i
    for i, panel_i in enumerate(panel_list):
        
        # for every other panel j
        for j, panel_j in enumerate(panel_list):
            
            # It cannot effect itself:
            if i == j:
                A[i, j] = 0
            
            else:
                A[i, j] = 1 / (2 * np.pi) * I(panel_i.xc,
                                              panel_i.yc,
                                              panel_j,
                                              -1 * np.sin(panel_i.beta),
                                              np.cos(panel_i.beta))
    # Find output vector b
    b = np.zeros((N, 1))
    for i, panel in enumerate(panel_list):
        b[i] = freestream.u_inf * np.sin(freestream.alpha
                                         - panel.beta)
    strength_list = np.zeros((N, 1))
    for i, panel in enumerate(panel_list):
        strength_list[i] = panel.strength
    
    v_tan = np.dot(A, strength_list) + b
    
    for i, panel in enumerate(panel_list):
        panel.v_tan = v_tan[i]
    
    return panel_list
    
def PanelPressureSolver(freestream, panel_list):
    
    for i, panel in enumerate(panel_list):
        panel.surface_cp = 1 - (panel.v_tan / freestream.u_inf)**2
        
    return panel_list

def ConstructAirfoilPanels(xpoints, ypoints):
    N = len(xpoints) - 1
    panel_list = np.empty((N), dtype=object)
    # print("Airfoil Panel Points:")
    for i in range(N):
        # print(xpoints[i], ypoints[i
        panel_list[i] = Panel(xpoints[i], ypoints[i], xpoints[i+1], ypoints[i+1])
    # print(xpoints[i+1], ypoints[i+1])
    
    return panel_list

def NACAThicknessEquation(N, A, CA, num_points, *, use_other_x_points=0):
    """
    Generates a non-dimensionalized NACA airfoil given NACA numbers.

    Parameters
    ----------
    N : Ratio of max camber to chord length
    A : Location of max camber
    CA : Thickness ratio 
    num_points : Number of points for airfoil


    Returns
    -------
    x_non_dim_full : List of non-dimenionalized points from 0 to 1 to 0
    z : Airfoil non-dimensionalized z poisition from xc = 0 to 1 to 0
    zcc : Chord line

    """
    p = 0.1 * A
    m = 0.01 * N
    t = 0.01 * CA
    if use_other_x_points is not 0:
        x_non_dim = use_other_x_points
    else:
        x_non_dim = np.linspace(0, 1, num_points)
    
    ztc = x_non_dim*0
    
    # Find thickness relative to camber
    ztc += 0.2969 * (x_non_dim**0.5)
    ztc -= 0.1260 * (x_non_dim**1)
    ztc -= 0.3516 * (x_non_dim**2)
    ztc += 0.2843 * (x_non_dim**3)
    ztc -= 0.1015 * (x_non_dim**4)
    
    ztc *= t/0.2
    
    
    # Find camber line
    zcc = 0*x_non_dim
    try:
        for i in zip(*np.where(x_non_dim <= p)):
            zcc[i] = 2*p*x_non_dim[i]
            zcc[i] -= x_non_dim[i]**2
            zcc[i] *= m * p**-2
    
        for i in zip(*np.where(x_non_dim > p)):
            zcc[i] = 1 - 2*p
            zcc[i] += 2*p*x_non_dim[i]
            zcc[i] -= x_non_dim[i]**2
            zcc[i] *= m * (1-p)**-2

    except:
        zcc = 0*x_non_dim


    # Sum the two
    zup = zcc + ztc
    zdown = zcc - ztc
    
    x_non_dim = np.concatenate((x_non_dim, np.flip(x_non_dim)[1:]))
    z = np.concatenate((zup, np.flip(zdown)[1:]))
    return x_non_dim, z

def I(x, y, panel, dxdz, dydz):

    integral_internal = lambda s: (((x - (panel.x1 - np.sin(panel.beta) * s)) * dxdz +
                                    (y - (panel.y1 + np.cos(panel.beta) * s)) * dydz) /
                                   (   (x - (panel.x1 - np.sin(panel.beta) * s))**2 +
                                    (y - (panel.y1 + np.cos(panel.beta) * s))**2) )
    return integrate.quad(integral_internal, 0.0, panel.length)[0]

def CalculateCn(Cplx, Cply, Cpux, Cpuy, *, c=1):
    left = np.trapz(Cply, Cplx)
    right = np.trapz(Cpuy, Cpux)
    out = (left - right)
    out /= c
    return out

def CalculateCa(Cplx, Cply, Cpux, Cpuy, xc, ztu, ztl,  *, c=1):
    gradu = np.gradient(ztu, xc)
    gradl = np.gradient(ztl, xc)
    
    graduvec = np.zeros((len(Cpux)))
    gradlvec = np.zeros((len(Cplx)))
    
    for i in range(len(graduvec)):
        graduvec[i] = np.interp(Cpux[i], xc, gradu)

    for i in range(len(gradlvec)):
        gradlvec[i] = np.interp(Cplx[i], xc, gradl)
    left = np.trapz(Cpuy*graduvec, Cpux)
    right = np.trapz(Cply*gradlvec, Cplx)

    
    out = (left - right)/c
    return out

def CalculateCmLE(Cplx, Cply, Cpux, Cpuy, *, c=1):
    """
    Determines moment coefficient from the leading edge.

    Parameters
    ----------
    Cplx : x-data for lower Cp
    Cply : y-data for lower Cp
    Cpux : x-data for upper Cp
    Cpuy : y-data for upper Cp
    c : Chord length, default 1 (non-dimensionalized)

    Returns
    -------
    CmLE : Moment Coefficient CmLE

    """
    left = np.trapz(Cpuy*Cpux, Cpux)
    right = np.trapz(Cply*Cplx, Cplx)
    CmLE = left - right
    CmLE /= c**2
    return CmLE

def CalculateVelocityField(X, Y, fs, panel_list):
    u = fs.u_inf * np.cos(fs.alpha) + X*0
    v = fs.u_inf * np.sin(fs.alpha) + X*0
    
    vector_I = np.vectorize(I)
    for panel in panel_list:
        u += panel.strength / (2*np.pi) * vector_I(X, Y, panel, 1, 0)
        v += panel.strength / (2*np.pi) * vector_I(X, Y, panel, 0, 1)
    
    return u, v

def CalculatePressureField(u, v, fs):
    Cp = 1 - (u**2 + v**2) / fs.u_inf**2
    return Cp

def plothusly(ax, x, y, *, xtitle='', ytitle='',
              datalabel='', title='', linestyle='-',
              marker=''):
    """
    A little function to make graphing less of a pain.
    Creates a plot with titles and axis labels.
    Adds a new line to a blank figure and labels it.

    Parameters
    ----------
    ax : The graph object
    x : X axis data
    y : Y axis data
    xtitle : Optional x axis data title. The default is ''.
    ytitle : Optional y axis data title. The default is ''.
    datalabel : Optional label for data. The default is ''.
    title : Graph Title. The default is ''.

    Returns
    -------
    out : Resultant graph.

    """

    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    ax.set_title(title)
    out = ax.plot(x, y, zorder=1, label=datalabel, linestyle = linestyle,
                  marker = marker)
    plt.grid()
    plt.legend(loc='best')
    return out

def plothus(ax, x, y, *, datalabel='', linestyle = '-',
            marker = ''):
    """
    A little function to make graphing less of a pain

    Adds a new line to a blank figure and labels it
    """
    out = ax.plot(x, y, zorder=1, label=datalabel, linestyle = linestyle,
                  marker = marker)
    plt.legend(loc='best')

    return out

#%%###########################

# Main
Nlist = [8, 32, 128]
fs = Freestream(1, 0)

# Create plot and add analytical solution. Can then add surface pressures for 
# each as part of a loop

fig, presplot = plt.subplots(figsize=(10, 5))

thetavec_analytical = np.linspace(-np.pi, np.pi, 1000)
cp_analytical = 1 - (-2*fs.u_inf*np.sin(thetavec_analytical) / fs.u_inf**2)**2
plothusly(presplot, 
          thetavec_analytical, 
          cp_analytical,
          title='Pressure Plot',
          xtitle=r'$x$',
          ytitle=r'C$_P$',
          datalabel=r'$C_P$ analytical')



for N in Nlist:
    
    # Construct Cylinder panels
    x_cylinder, y_cylinder, panel_list = ConstructCylinder(0, 0, 1, N)
    
    # Solve for all data
    panel_list = PanelStrengthSolver(fs, panel_list)
    panel_list = PanelTangentSolver(fs, panel_list)
    panel_list = PanelPressureSolver(fs, panel_list)
    
    # Plot discretized pressure
    
    thetavec = np.empty((N, 1))
    cpvec = np.empty((N, 1))
    for i, panel in enumerate(panel_list):
        thetavec[i] = np.arctan2(panel.yc, panel.xc)
        cpvec[i] = panel.surface_cp
    plothus(presplot, thetavec, cpvec, datalabel=fr'Panel method, $n$ = {N} ', linestyle='', marker='o')
    
    datavec = np.concatenate((thetavec, cpvec), axis=1)
    datavec = datavec[datavec[:, 0].argsort()]
    
    # Calculate error
    err = (np.trapz(cp_analytical,
                    thetavec_analytical)
           - np.trapz(datavec[:, 1],
                      datavec[:, 0]))
    
    print(f'Error for {N} terms is {err}')
    
    
    # Check to ensure that conservation of mass is true
    sumval = 0
    for i, panel in enumerate(panel_list):
        sumval += panel.strength * panel.length
        
    print(f'Sanity Check for N = {N} : {sumval}')
    
    
#%%###########################

# Special Case for 32 Panels

N = 32

# Construct Cylinder panels
x_cylinder, y_cylinder, panel_list = ConstructCylinder(0, 0, 1, N)
    
# Solve for all data
panel_list = PanelStrengthSolver(fs, panel_list)
panel_list = PanelTangentSolver(fs, panel_list)
panel_list = PanelPressureSolver(fs, panel_list)
    
# Find flowfield

# meshgrid

x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)

X, Y = np.meshgrid(x, y)

u, v = CalculateVelocityField(X, Y, fs, panel_list)    

Cp = CalculatePressureField(u, v, fs)
# Plot it

fig, streamplot = plt.subplots()

plt.streamplot(X, Y, u, v, density = 2,
               linewidth = 1,
               color='blue',
               cmap=None,
               arrowsize=1)

plt.fill([panel.xc for panel in panel_list],
         [panel.yc for panel in panel_list],
         color='black',
         linestyle='-',
         linewidth=2,
         zorder=2)

plt.contourf(X, Y, Cp, cmap='viridis')

plt.axis('equal')

plt.colorbar()

#%%###########################

thetavec = np.empty((N, 1))
cpvec = np.empty((N, 1))
for i, panel in enumerate(panel_list):
    thetavec[i] = np.arctan2(panel.yc, panel.xc)
    cpvec[i] = panel.surface_cp
    
datavec = np.concatenate((thetavec, cpvec), axis=1)
datavec = datavec[datavec[:, 0].argsort()]

datavec[:, 1] = np.gradient(datavec[:, 1], datavec[:, 0])

fig, gradplot = plt.subplots()
plothusly(gradplot, datavec[:, 0], datavec[:, 1], xtitle=r'$\theta$', 
          ytitle=r'Pressure Gradient $\frac{\partial C_P}{\partial \theta}$',
          datalabel='32 Panels',
          title="Pressure Gradient for 32 Panel Cylinder",
          marker='x')
#%%###########################

# Log-log plot

Nlist = [8, 16, 32, 64, 128, 256, 512, 1024]
errdat = np.empty((len(Nlist)))

for i, N in enumerate(Nlist):
    
    # Construct Cylinder panels
    x_cylinder, y_cylinder, panel_list = ConstructCylinder(0, 0, 1, N)
    
    # Solve for all data
    panel_list = PanelStrengthSolver(fs, panel_list)
    panel_list = PanelTangentSolver(fs, panel_list)
    panel_list = PanelPressureSolver(fs, panel_list)
    
    # Plot discretized pressure
    
    thetavec = np.empty((N, 1))
    cpvec = np.empty((N, 1))    

    for j, panel in enumerate(panel_list):
        thetavec[j] = np.arctan2(panel.yc, panel.xc)
        cpvec[j] = panel.surface_cp

    datavec = np.concatenate((thetavec, cpvec), axis=1)
    datavec = datavec[datavec[:, 0].argsort()]

    # Calculate error
    errdat[i] = (np.trapz(cp_analytical,
                    thetavec_analytical)
           - np.trapz(datavec[:, 1],
                      datavec[:, 0]))


    print("Ding")
fig, errplot = plt.subplots()

plothusly(errplot, Nlist, abs(errdat), xtitle='Number of Panels', ytitle='Error from analytical',
          title='Panel Method Error for cylinders of varing panel sizes',
          datalabel='Error vs. Analytical')

plt.yscale('log')
plt.xscale('log')

#%%###########################

N = 102
xreal, zreal = NACAThicknessEquation(0, 0, 18, 1000)
x_panel_points, z_panel_points = NACAThicknessEquation(0, 0, 18, N)


# Fixes "Inside Out" problem
x_panel_points, z_panel_points = np.flip(x_panel_points), np.flip(z_panel_points)

panel_list = ConstructAirfoilPanels(x_panel_points, z_panel_points)

# Create plot and show the discretized airfoil vs the real one


panel_list = PanelStrengthSolver(fs, panel_list)
panel_list = PanelTangentSolver(fs, panel_list)
panel_list = PanelPressureSolver(fs, panel_list)
    
    
fig, foilplot = plt.subplots()

plothusly(foilplot, 
          xreal,
          zreal,
          xtitle='x',
          ytitle='y',
          title='Airfoil Comparison',
          datalabel='NACA 0018 Geometry')

plothus(foilplot,
        x_panel_points,
        z_panel_points,
        datalabel='Discretized Points',
        marker='o',
        linestyle='--')
plt.axis('equal')

sumval = 0
for i, panel in enumerate(panel_list):
    sumval += panel.strength * panel.length
    
    
# Get Data from XFOIL

airfoilflatLOlo = np.loadtxt("Data/naca0018ReLO0lower.text", skiprows=1)
airfoilflatLOhi = np.loadtxt("Data/naca0018ReLO0lower.text", skiprows=1)



print(f'Sanity Check for NACA 0018 : {sumval}')

pressure_list_upper = [panel.surface_cp.item() for panel in panel_list if panel.loc == 'upper']
pressure_x_upper = [panel.xc for panel in panel_list if panel.loc == 'upper']

fig, airfoilpressure = plt.subplots()
plothusly(airfoilpressure, 
          pressure_x_upper,
          pressure_list_upper,
          datalabel='Panel Method, Upper Surface',
          xtitle=r"Non-dimensionalized Chord length $\frac{x}{c}$",
          ytitle=r'$C_P$',
          title=fr'NACA 0018 Pressure Distribution, N = {len(panel_list)}',
          marker='x')

pressure_list_lower = [panel.surface_cp.item() for panel in panel_list if panel.loc == 'lower']
pressure_x_lower = [panel.xc for panel in panel_list if panel.loc == 'lower']

CL = CalculateCn(pressure_x_lower,
                 pressure_list_lower,
                 pressure_x_upper,
                 pressure_list_upper)

CD = CalculateCn(pressure_x_lower,
                 pressure_list_lower,
                 pressure_x_upper,
                 pressure_list_upper)

plt.xlim([0, 1])
plt.gca().invert_yaxis()

print(CL, CD)
plothus(airfoilpressure,
        pressure_x_lower,
        pressure_list_lower,
        datalabel='Panel Method, Lower Surface',
        marker='*')


plothus(airfoilpressure,
        airfoilflatLOhi[:, 0],
        airfoilflatLOhi[:, 1],
        datalabel='XFOIL, Upper surface',
        )


plothus(airfoilpressure,
        airfoilflatLOlo[:, 0],
        airfoilflatLOlo[:, 1],
        datalabel='XFOIL, Lower surface',
        )

#%%###########################

x = np.linspace(-1.5, 1.5, 50)
y = np.linspace(-2, 2, 50)

X, Y = np.meshgrid(x, y)

u, v = CalculateVelocityField(X, Y, fs, panel_list)    

# Plot it

fig, streamplot = plt.subplots()

plt.streamplot(X, Y, u, v, density = 2,
               linewidth = 1,
               color='blue',
               cmap=None,
               arrowsize=1)

plt.fill([panel.xc for panel in panel_list],
         [panel.yc for panel in panel_list],
         color='black',
         linestyle='-',
         linewidth=2,
         zorder=2)

plt.axis('equal')
plt.colorbar()


#%%###########################

# Force Coefficients

# Nlist = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 160, 200, 400, 600, 1000]

Nlist = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
Cl = np.zeros((len(Nlist)))
Cd = np.zeros((len(Nlist)))

for i, N in enumerate(Nlist):
    x_panel_points, z_panel_points = NACAThicknessEquation(0, 0, 18, N)


    # Fixes "Inside Out" problem
    x_panel_points, z_panel_points = np.flip(x_panel_points), np.flip(z_panel_points)

    panel_list = ConstructAirfoilPanels(x_panel_points, z_panel_points)

    # Solve for panel data
    

    panel_list = PanelStrengthSolver(fs, panel_list)
    panel_list = PanelTangentSolver(fs, panel_list)
    panel_list = PanelPressureSolver(fs, panel_list)
    
    pressure_list_upper = [panel.surface_cp.item() for panel in panel_list if panel.loc == 'upper']
    pressure_x_upper = [panel.xc for panel in panel_list if panel.loc == 'upper']
    pressure_y_upper = [panel.yc for panel in panel_list if panel.loc == 'upper']
    
    pressure_list_lower = [panel.surface_cp.item() for panel in panel_list if panel.loc == 'lower']
    pressure_x_lower = [panel.xc for panel in panel_list if panel.loc == 'lower']
    pressure_y_lower = [panel.yc for panel in panel_list if panel.loc == 'lower']


    Cl[i] = CalculateCn(pressure_x_lower,
                        pressure_list_lower,
                        np.flip(pressure_x_upper),
                        np.flip(pressure_list_upper))

    Cd[i] = CalculateCa(pressure_x_lower,
                        pressure_list_lower,
                        pressure_x_upper,
                        pressure_list_upper,
                        pressure_x_lower, 
                        pressure_y_lower,
                        np.flip(pressure_y_upper))
    print(f"Ding {N}")

# Create semilog plot

fig, liftplot = plt.subplots()

plothusly(liftplot, Nlist, Cl, datalabel=r'$C_l$', xtitle='Number of panels',
          ytitle='Coefficient Value', title='Force Coefficients for increasing panel number',
          marker='x')

plothus(liftplot, Nlist, Cd, datalabel=r'$C_d$', marker='+')
plt.xscale('log')

#%%###########################

# Problem 3.4

Nlist = [128]


for N in Nlist:
    
    # Construct Cylinder panels
    x_cylinder, y_cylinder, panel_list = ConstructCylinder(0, 0, 1, N)
    
    # Solve for all data
    panel_list = PanelStrengthSolver(fs, panel_list)
    panel_list = PanelTangentSolver(fs, panel_list)
    panel_list = PanelPressureSolver(fs, panel_list)
    
    # Plot discretized pressure
    
    thetavec = np.empty((N, 1))
    cpvec = np.empty((N, 1))
    for i, panel in enumerate(panel_list):
        thetavec[i] = np.arctan2(panel.yc, panel.xc)
        cpvec[i] = panel.surface_cp
    plothus(presplot, thetavec, cpvec, datalabel=fr'Panel method, $n$ = {N} ', linestyle='', marker='o')
    
    datavec = np.concatenate((thetavec, cpvec), axis=1)
    datavec = datavec[datavec[:, 0].argsort()]

Pamb = 2116.23  # lbf/ft**2
rho = 0.00237717  # slug/ft**3
v_inf = 312.245  # ft/s

Pfind = lambda Cp: ((Cp + Pamb) * (0.5 * rho * v_inf**2))

Cp0 = np.interp(0, datavec[:, 0], datavec[:, 1])
Cp45 = np.interp(np.pi/4, datavec[:, 0], datavec[:, 1])

print(f'Pressure at 0 deg is {Pfind(Cp0)} lbf/sqft')
print(f' Pressure at 0 deg is {Pfind(Cp45)} lbf/sqft')



