#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 09:46:41 2020

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
plt.style.use("seaborn-bright")

#%%###########################

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


class Source():
    
    def __init__(self, x, y, strength):
        """
        Generates a source/sink at given coordinates, with a given strength.
        Positive strength indiates source, negative strenght indicates sink.

        Parameters
        ----------
        x : X coordinate of source
        y : Y coordinate of source
        strength : Capital lambda, source strength

        Returns
        -------
        None.

        """
        
        self.pos = np.array([[x],
                             [y]])
        self.strength = strength
        
    
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
        psi_mesh = self.strength / (2*np.pi) * np.arctan2((Ymesh - self.pos[1]),
                                                         (Xmesh - self.pos[0]))
        return psi_mesh
    
    def GetStreamPotential(self, Xmesh, Ymesh):
        
        phi_mesh = self.strength / (2*np.pi) * np.log(((Xmesh - self.pos[0])**2 + (Ymesh - self.pos[1])**2)**0.5)

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
        umesh = self.strength / (2*np.pi)
        umesh *= (Xmesh - self.pos[0])
        umesh /= ((Xmesh - self.pos[0])**2 + (Ymesh - self.pos[1])**2)
        
        vmesh = self.strength / (2*np.pi)
        vmesh *= (Ymesh - self.pos[1])
        vmesh /= ((Xmesh - self.pos[0])**2 + (Ymesh - self.pos[1])**2)
        
        return umesh, vmesh

        

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
        self.cp = 0
        
        
        if x2 -x1 <= 0:
            self.beta = np.arccos((y2 - y1) / self.length)
        elif x2 - x1 >= 0:
            self.beta = np.pi + np.arccos(-(y2 - y1) / self.length)
        
    
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

    # def Integral(self, x, y, dxdz, dydz):
        
    #     integrand = lambda s: (((x - (self.x1 - np.sin(self.beta) * s)) * dxdz +
    #                             (y - (self.y1 + np.cos(self.beta) * s)) * dydz) /
    #                            ((x - (self.x1 - np.sin(self.beta) * s))**2 +
    #                             (y - (self.y1 + np.cos(self.beta) * s))**2) )

    #     return integrate.quad(integrand, 0.0, self.length)[0]
    
#%%###########################

# Functions


def CreateStreamPlot(obj_list, X, Y, ShowCP=False):
    """
    

    Parameters
    ----------
    source_x_list : List of source x-coords
    source_y_list : List of source y-coords
    source_l_list : List of source strengths
    u_inf : Freestream U
    v_inf : Freestream V
    X : X Meshgrid
    Y : Y Meshgrid

    Returns
    -------
    streamplot : Plot of superimposed fluid flow

    """
    # Creates empty list to store source objects
    n = len(obj_list)
    N = len(X)

    # Creates Total sum field
    u_total = np.zeros((N, N))
    v_total = np.zeros((N, N))
    psi_total = np.zeros((N, N))
    phi_total = np.zeros((N, N))


    # Adds sources and sinks to total field
    for i, obj in enumerate(obj_list):
        if isinstance(obj, Freestream) is True:
            freestream = obj
            u_inf = freestream.u_inf
            u_source , v_source = obj.GetVelocity(X, Y)
            u_source , v_source = obj.GetVelocity(X, Y)
            psi_source = obj.GetStreamFunction(X, Y)
        else:
            u_source , v_source = obj.GetVelocity(X, Y)
            u_source , v_source = obj.GetVelocity(X, Y)
            psi_source = obj.GetStreamFunction(X, Y)

        u_source , v_source = obj.GetVelocity(X, Y)
        psi_source = obj.GetStreamFunction(X, Y)
        phi_source = obj.GetStreamPotential(X, Y)

        u_total += u_source
        v_total += v_source
        psi_total += psi_source
        phi_total += phi_source


    # Creates streamline plot
    fig, streamplot = plt.subplots()
    plt.grid(True)
    plt.xlabel("X Direction")
    plt.ylabel("Y Direction")
    plt.streamplot(X, Y, u_total, v_total, 
                    density = 2.5, linewidth = 0.4, arrowsize=0.8,
                    )
    plt.title("Stream Plot")
    
    plt.contour(X, Y, psi_total, levels=[0.], linewidths=2, colors='red', linestyles='dashdot', alpha=1)
        
    # Potential
    plt.contour(X, Y, phi_total, colors='black', linestyles='--')
    plt.legend(loc='upper left')
    # plt.axis('equal')
    
    
    # Pressure
    if ShowCP == True:
        
        Cp = 1 - ((u_total**2 + v_total**2) / u_inf**2)
        plt.contourf(X, Y, Cp, cmap='coolwarm')
        cbar = plt.colorbar()
        cbar.set_label(r'$C_P$')
    else:
        pass
    
    return streamplot

def ConstructCylinder(xc, yc, R, N):
    
    xpoints = R * np.cos(np.linspace(0, 2*np.pi, N+1))
    ypoints = R * np.sin(np.linspace(0, 2*np.pi, N+1))
    
    panel_list = np.empty((N), dtype=object)
    for i in range(N):
        panel_list[i] = Panel(xpoints[i], ypoints[i], xpoints[i+1], ypoints[i+1])
    

    return xpoints, ypoints, panel_list


def PanelStrengthSolver(freestream, panel_list):
    N = len(panel_list)

    A = np.empty((N,N))
    
    for i, panel_i in enumerate(panel_list):
        
        # for every other panel j:
        for j, panel_j in enumerate(panel_list):


            # If it equals itself, use 1/2
            if i == j:
                A[i, j] = 0.5
            
            # Otherwise use normal integral
            else:
                A[i, j] = 0.5/np.pi* NormalIntegral(panel_i,
                                                    panel_j)

    # Create solution vector
    solvec = np.zeros((N, 1))
    for i in range(N):
        solvec[i] = -freestream.u_inf * np.cos(panel_list[i].beta)
    
    # 
    strength_list = np.linalg.solve(A, solvec)
    for i, panel in enumerate(panel_list):
        panel.strength = strength_list[i]
    
    # return updated panel list
    return panel_list



def AirfoilPanelStrengthSolver(freestream, panel_list):
    N = len(panel_list)
    A = np.empty((N,N))
    
    for i, panel_i in enumerate(panel_list):
        
        # for every other panel j:
        for j, panel_j in enumerate(panel_list):


            # If it equals itself, use 1/2
            if i == j:
                A[i, j] = 0.5
            
            # Otherwise use normal integral
            else:
                A[i, j] = 0.5/np.pi* AirfoilIntegral(panel_i.xc, 
                                                     panel_i.yc,
                                                     panel_j,
                                                     np.cos(panel_i.beta),
                                                     np.sin(panel_i.beta)
                                                     )

    # Create solution vector
    solvec = np.zeros((N, 1))
    for i, panel in enumerate(panel_list):
        solvec[i] = -freestream.u_inf * np.cos(panel.beta - freestream.alpha)

    strength_list = np.linalg.solve(A, solvec)
    for i, panel in enumerate(panel_list):
        panel.strength = strength_list[i]
    
    # return updated panel list
    return panel_list


def CPfinder(freestream, panel_list, N):
    A = np.empty((N, N))
    
    for i, panel_i in enumerate(panel_list):
        for j, panel_j in enumerate(panel_list):
            
            # Condition if panel applied to self
            if i == j:
                A[i, j] = 0
                
            else:
                A[i, j] = 0.5 / np.pi * TangentialIntegral(panel_i,
                                                           panel_j)
    
    solvec = np.zeros((N, 1))
    strength_list = np.zeros((N, 1))
    for i, panel in enumerate(panel_list):
        solvec[i] = -freestream.u_inf * np.sin(panel.beta - freestream.alpha)
        strength_list[i] = panel.strength
    
    v_tan_list = np.dot(A,strength_list) + solvec
    
    for i, panel in enumerate(panel_list):
        panel.v_tan = v_tan_list[i]
    
    for panel in panel_list:
        panel.cp = 1 - (panel.v_tan / freestream.u_inf)**2
    
    return panel_list

    
def NormalIntegral(p_i, p_j):
    def integrand(s):
        return (((p_i.xc - (p_j.x1 - np.sin(p_j.beta) * s)) * np.cos(p_i.beta) +
                 (p_i.yc - (p_j.y1 + np.cos(p_j.beta) * s)) * np.sin(p_i.beta)) /
                ((p_i.xc - (p_j.x1 - np.sin(p_j.beta) * s))**2 +
                 (p_i.yc - (p_j.y1 + np.cos(p_j.beta) * s))**2))
    return integrate.quad(integrand, 0.0, p_j.length)[0]


def TangentialIntegral(p_i, p_j):
    
    tan_int = lambda s: ((-(p_i.xc - (p_j.x1 - np.sin(p_j.beta) * s)) * np.sin(p_i.beta) +
                 (p_i.yc - (p_j.y1 + np.cos(p_j.beta) * s)) * np.cos(p_i.beta)) /
                ((p_i.xc - (p_j.x1 - np.sin(p_j.beta) * s))**2 +
                 (p_i.yc - (p_j.y1 + np.cos(p_j.beta) * s))**2))
    return integrate.quad(tan_int, 0.0, p_j.length, limit=100)[0]


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

def AirfoilIntegral(x, y, panel, dxdz, dydz):
    
    panel_cont = lambda s: (((x - (panel.x1 - np.sin(panel.beta) * s)) * dxdz +
                 (y - (panel.y1 + np.cos(panel.beta) * s)) * dydz) /
                ((x - (panel.x1 - np.sin(panel.beta) * s))**2 +
                 (y - (panel.y1 + np.cos(panel.beta) * s))**2) )
    
    return integrate.quad(panel_cont, 0.0, panel.length)[0]

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
xc = 0
yc = 0
R = 30
Nlist = [8, 32]
u_inf = 1
freestream = Freestream(u_inf, 0)

fig, presplot = plt.subplots(figsize=(16, 8))

xcyl, ycyl, panel_list = ConstructCylinder(xc, yc, R, 10)
thetavec = np.linspace(-np.pi, np.pi, 1000)
cp_analytical = 1 - (-2*freestream.u_inf*np.sin(thetavec) / freestream.u_inf**2)**2
plothusly(presplot, thetavec, cp_analytical,title='Pressure Plot', xtitle=r'$x$', ytitle=r'C$_P$',  datalabel=r'$C_P$ analytical')
plt.ylim([-3, 1])



for N in Nlist:
    xcyl, ycyl, panel_list = ConstructCylinder(xc, yc, R, N)
    
    panel_list = PanelStrengthSolver(freestream, panel_list)
    panel_list = CPfinder(freestream, panel_list, N)
    thetavec = np.empty((N, 1))
    cpvec = np.empty((N, 1))
    for i, panel in enumerate(panel_list):
        thetavec[i] = np.arctan2(panel.yc, panel.xc)
        cpvec[i] = panel.cp
    plothus(presplot, thetavec, cpvec, datalabel=fr'Panel method, $n$ = {N} ', linestyle='', marker='o')

    if N == 32:
        objlist = np.empty((1), dtype = object)
        objlist[0] = freestream
        objlist = np.concatenate((objlist, panel_list))
        # Make list of objects, then add together.
        xdat = np.linspace(-1, 1, 10)
        ydat = np.linspace(-1, 1, 10)
        X, Y = np.meshgrid(xdat, ydat)
        CreateStreamPlot(objlist, X, Y)
        # ydat = np.linspace(-3, 3, 200)
        # X, Y = np.meshgrid(xdat, ydat)
        # r = (X**2 + Y**2)**0.5
        # theta = np.arctan2(Y, X)
        # v_theta = -1*(1 + (R**2)/(r**2)) * u_inf * np.sin(theta)
        # CpGrid = 1 - (v_theta / (u_inf**2))
        # fig, pressureplot = plt.subplots()
        # plt.contourf(X, Y, v_theta, cmap='viridis',
        #               levels=np.linspace(-2, 2, 255))
        # plt.plot(xcyl, ycyl)
        # plt.colorbar()

sumval = 0
for i, panel in enumerate(panel_list):
    sumval += panel.strength * panel.length

print(f'Sanity Check = {sumval}')
#%%###########################

fig, ax = plt.subplots()

N = 120
# Make Points, linearly spaced in x dir
xchord, ychord = NACAThicknessEquation(0, 0, 18, N+1)
xreal, yreal = NACAThicknessEquation(0, 0, 18, 10000)

# I don't know why but this fixes a bug
xchord, ychord = np.flip(xchord), np.flip(ychord)


panel_list = np.empty((N*2), dtype=object)
xcoords = np.zeros((N*2))
ycoords = np.zeros((N*2))

for i in range(2*N):
    panel_list[i] = Panel(xchord[i], ychord[i], xchord[i+1], ychord[i+1])

panel_list = AirfoilPanelStrengthSolver(freestream, panel_list)


objlist 
CreateStreamPlot(obj_list, X, Y)

plothus(ax, xchord, ychord, marker='o')
plt.axis('equal')

# for i, panel in enumerate(panel_list):
#     print(panel.beta)

sumval = 0
for i, panel in enumerate(panel_list):
    sumval += panel.strength * panel.length

print(f'Sanity Check = {sumval}')
#%%###########################
