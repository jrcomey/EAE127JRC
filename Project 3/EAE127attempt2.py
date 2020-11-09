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
        
        if y2 - y1 <= 0:
            self.beta = np.arcsin((x2 - x1) / self.length)
        elif y2 - y1 >= 0:
            self.beta = np.pi + np.arcsin(-(x2 - x1) / self.length)
        
    
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
                 A[i, j] = 1 / (2*np.pi) * integral(panel_i.xc,
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
                A[i, j] = 1 / (2 * np.pi) * integral(panel_i.xc,
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

def integral(x, y, panel, dxdz, dydz):

    integral_internal = lambda s: (((x - (panel.x1 - np.sin(panel.beta) * s)) * dxdz +
                 (y - (panel.y1 + np.cos(panel.beta) * s)) * dydz) /
                ((x - (panel.x1 - np.sin(panel.beta) * s))**2 +
                 (y - (panel.y1 + np.cos(panel.beta) * s))**2) )
    return integrate.quad(integral_internal, 0.0, panel.length)[0]

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

#%###########################

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
    
    thetavec = np.empty((N))
    cpvec = np.empty((N))
    for i, panel in enumerate(panel_list):
        thetavec[i] = np.arctan2(panel.yc, panel.xc)
        cpvec[i] = panel.surface_cp
    plothus(presplot, thetavec, cpvec, datalabel=fr'Panel method, $n$ = {N} ', linestyle='', marker='o')
    
    # Calculate error
    err = (np.trapz(cp_analytical,
                    thetavec_analytical)
           - np.trapz(cpvec,
                      thetavec))
    
    print(f'Error for {N} terms is {err}')
    
    
    # Check to ensure that conservation of mass is true
    sumval = 0
    for i, panel in enumerate(panel_list):
        sumval += panel.strength * panel.length
        
    print(f'Sanity Check for N = {N} : {sumval}')


#%%###########################

xreal, zreal = NACAThicknessEquation(2, 4, 18, 1000)
x_panel_points, z_panel_points = NACAThicknessEquation(2, 4, 18, 100)


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

sumval = 0
for i, panel in enumerate(panel_list):
    sumval += panel.strength * panel.length

print(f'Sanity Check for NACA 0018 : {sumval}')
    

plt.axis('equal')