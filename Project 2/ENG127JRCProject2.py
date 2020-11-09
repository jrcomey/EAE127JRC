#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:57:10 2020

@author: jack

EAE 127 Project 2

Stream sources and sink superposition project
"""

# Imports 

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("classic")
#%%###########################

# Objects

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

        
#%###########################

# Functions


def CreateStreamPlot(source_x_list, source_y_list, source_l_list, u_inf, v_inf, X, Y, N):
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
    n = len(source_x_list)
    source_sink_objs = np.empty((n, 1), dtype=object)
    
    
    # Creates the source objects given input properties and stores
    # them in the list
    i = 0
    for x, y, l in zip(source_x_list, source_y_list, source_l_list):
        source_sink_objs[i] = Source(x, y, l)
        i += 1
    
    # Creates freestream velocity field
    u_freestream = u_inf * np.ones((N, N))
    v_freestream = v_inf * np.ones((N, N))
    psi_freestream = u_inf * Y
    phi_freestream = u_inf * X

    # Creates Total sum field
    u_total = np.zeros((N, N))
    v_total = np.zeros((N, N))
    psi_total = np.zeros((N, N))
    phi_total = np.zeros((N, N))
    
    # Adds freestream to total
    u_total += u_freestream
    v_total += v_freestream
    psi_total += psi_freestream
    phi_total += phi_freestream


    # Adds sources and sinks to total field
    for i in range(len(source_sink_objs)):
        u_source , v_source = source_sink_objs[i].item().GetVelocity(X, Y)
        psi_source = source_sink_objs[i].item().GetStreamFunction(X, Y)
        phi_source = source_sink_objs[i].item().GetStreamPotential(X, Y)
        u_total += u_source
        v_total += v_source
        psi_total += psi_source
        phi_total += phi_source


    # Creates a list of sources and a list of sinks from inputs
    # This works, don't touch it
    x_source = np.zeros((len(np.where(source_l_list > 0)[0]), 1))
    y_source= np.zeros((len(np.where(source_l_list > 0)[0]), 1))
    x_sink = np.zeros((len(np.where(source_l_list < 0)[0]), 1))
    y_sink = np.zeros((len(np.where(source_l_list < 0)[0]), 1))

    # Creates list of source coordinates
    k = 0
    for i in zip(*np.where(source_l_list > 0)):
        x_source[k], y_source[k] = source_x_list[i], source_y_list[i]
        k += 1
    
    # Creates list of sink coordinates
    k = 0
    for i in zip(*np.where(source_l_list < 0)):
        x_sink[k], y_sink[k] = source_x_list[i], source_y_list[i]
        k += 1
    

    # Creates streamline plot
    fig, streamplot = plt.subplots()
    plt.grid(True)
    plt.xlabel("X Direction")
    plt.ylabel("Y Direction")
    plt.streamplot(X, Y, u_total, v_total, 
                    density = 2.5, linewidth = 0.4, arrowsize=0.8,
                    )
    plt.title("Stream Plot")
    plt.scatter(x_source, y_source, s=40, marker='o', color='red',label='Sources')
    plt.scatter(x_sink, y_sink, s=40, marker='o', color='black', label="Sinks")
    
    # For 1 source
    if len(source_l_list) < 2:
        
        # Contour
        plt.contour(X, Y, psi_total, levels=[-source_l_list[0]/2, source_l_list[0]/2], colors='red', linestyles='solid', alpha = 1)
        
        # Stagnation Point
        xstag = source_sink_objs[0].item().pos[0] - source_sink_objs[0].item().strength/(2*np.pi*u_inf)
        ystag = source_sink_objs[0].item().pos[1]
        plt.scatter(xstag, ystag, s=60, color='green', label='Stagnation Point')
        
        # Dividing SL diameter
        source_diameter_function = lambda Vinf, Lambda: (Lambda/(2*Vinf))
        divSL = source_diameter_function(u_inf, source_l_list[0])
        divSLdat = np.linspace(-divSL/2, divSL/2, 10)
        divSLxdat = divSLdat * 0
        plt.plot(divSLxdat, divSLdat, linewidth = 2, label=f"Dividing SL Diameter = {divSL:.2f} ft")
        
    # All other cases
    else:
        plt.contour(X, Y, psi_total, levels=[0.], linewidths=2, colors='red', linestyles='dashdot', alpha=1)
        
    # Potential
    plt.contour(X, Y, phi_total, colors='black', linestyles='--')
    plt.legend(loc='upper left')
    # plt.axis('equal')
    
    return streamplot

def ArbitraryErrorMetric(flow_x, flow_y, geo_x, geo_y):
    """
    Slight modification of the error metric as defined in the problem statement

    Parameters
    ----------
    flow_x : X coordinates for flow 
    flow_y : Y coordinates for flow 
    geo_x : X coordniates for airfoil geometry
    geo_y : Y coordinates for flow geometry

    Returns
    -------
    error : An error value normalized to the number of points used

    """
    
    error = 0
    # For every point taken from the flow:
    for x in flow_x:
        
        # Interpolate point from geometry
        flow_loc_z = flow_y[np.where(flow_x == x)]
        geo_interp_z = np.interp(x, geo_x, geo_y)
        
        error += (flow_loc_z - geo_y)**2
    
    error /= len(flow_x)
    
    return error
        

    
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
    
    x_non_dim = np.concatenate((x_non_dim, np.flip(x_non_dim)))
    z = np.concatenate((zup, np.flip(zdown)))
    return x_non_dim, z


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

source_x_list = np.array([0])
source_y_list = np.array([0])
source_l_list = np.array([2.2])

u_inf = 0.6
v_inf = 0

N = 1000

x_bounds = np.linspace(-4, 4, N)
y_bounds = np.linspace(-3, 3, N)

X, Y = np.meshgrid(x_bounds, y_bounds)


CreateStreamPlot(source_x_list, source_y_list, source_l_list, u_inf, v_inf, X, Y, N)
plt.axis('equal')

source_diameter_function = lambda Vinf, Lambda: (Lambda/(2*Vinf))

source_list = np.linspace(0, 10, 100)
dia_list = source_diameter_function(u_inf, source_list)

fig, diaplot = plt.subplots()
plothusly(diaplot, source_list, dia_list,
          xtitle=r"Source Strength $\Lambda$",
          ytitle=r"Dividing Streamline Diameter $D$",
          title=r"Dividing SL diameter as a function of $\Lambda$",
          datalabel="Dividing SL Diameter")


# #%%###########################


# source_x_list = np.array([0.1, 0.4, 0.45, 0.5, 0.8, 0.6, 0.65, 1, 2, 3])
# source_y_list = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# source_l_list = np.array([0.3, 0.05, 0.05, -0.05, 0.1, 0.05, -0.05, -0.1, -0.2, -0.15])

# u_inf = 0.5
# v_inf = 0

# N = 1000

# x_bounds = np.linspace(-1, 4, N)
# y_bounds = np.linspace(-2, 2, N)

# X, Y = np.meshgrid(x_bounds, y_bounds)


# airfoil_geometry = pd.DataFrame()

# x, zup = NACAThicknessEquation(0, 0, 20, 1000)

# airfoil_geometry["NACA 0020 X"], airfoil_geometry["NACA 0020"] = x, zup



# airfoil_geometry *= 3
# stream2 = CreateStreamPlot(source_x_list, source_y_list, source_l_list, u_inf, v_inf, X, Y, N)

# plt.xlim([-0.5, 3.5])
# plt.ylim([-0.5, 0.5])

# x *= 3
# zup *= 3
# ind = np.where(x < 3*0.75)
# x = x[ind]
# z = zup[ind]

# ind = np.where(z > 0)
# x = x[ind]
# z = z[ind]

# streamlinedat = stream2.collections[3].get_paths()[0].vertices
# streamlinedat = streamlinedat[np.where(streamlinedat[:, 1] > 0.0001)]
# streamlinedat = streamlinedat[np.where(streamlinedat[:, 0] > 0)]
# streamlinedat = streamlinedat[np.where(streamlinedat[:, 0] < 3*0.75)]

# a1 = np.trapz(z, x)
# a2 = np.trapz(streamlinedat[:, 1], streamlinedat[:, 0])

# arbitrary_error_metric = (a1 - a2)**2
# string = f'Arbitrary error metric value = {arbitrary_error_metric}'



# plothus(stream2, airfoil_geometry["NACA 0020 X"], airfoil_geometry["NACA 0020"],
#         datalabel='NACA 0020 Geometry')
# plt.axis('equal')

# #%%###########################

# source_x_list = np.array([0.1, 0.4, 0.45, 0.5, 0.8, 0.6, 0.65, 1.25, 2, 3])
# source_y_list = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# source_l_list = np.array([0.3, 0.05, 0.05, -0.05, 0.1, 0.05, -0.05, -0.2, -0.1, -0.15])

# u_inf = 0.5
# v_inf = 0

# N = 1000

# x_bounds = np.linspace(-1, 4, N)
# y_bounds = np.linspace(-2, 2, N)

# X, Y = np.meshgrid(x_bounds, y_bounds)


# airfoil_geometry = pd.DataFrame()

# x, zup = NACAThicknessEquation(0, 0, 20, 1000)

# airfoil_geometry["NACA 0020 X"], airfoil_geometry["NACA 0020"] = x, zup



# airfoil_geometry *= 3
# stream2 = CreateStreamPlot(source_x_list, source_y_list, source_l_list, u_inf, v_inf, X, Y, N)

# plt.xlim([-0.5, 3.5])
# plt.ylim([-0.5, 0.5])

# x *= 3
# zup *= 3
# ind = np.where(x < 3*0.75)
# x = x[ind]
# z = zup[ind]

# ind = np.where(z > 0)
# x = x[ind]
# z = z[ind]

# streamlinedat = stream2.collections[3].get_paths()[0].vertices
# streamlinedat = streamlinedat[np.where(streamlinedat[:, 1] > 0.0001)]
# streamlinedat = streamlinedat[np.where(streamlinedat[:, 0] > 0)]
# streamlinedat = streamlinedat[np.where(streamlinedat[:, 0] < 3*0.75)]

# a1 = np.trapz(z, x)
# a2 = np.trapz(streamlinedat[:, 1], streamlinedat[:, 0])

# arbitrary_error_metric = (a1 - a2)**2
# string = f'Arbitrary error metric value = {arbitrary_error_metric}'



# plothus(stream2, airfoil_geometry["NACA 0020 X"], airfoil_geometry["NACA 0020"],
#         datalabel='NACA 0020 Geometry')
# plt.axis('equal')

# #%%###########################

# source_x_list = np.array([0.1, 0.4, 0.45, 0.8, 0.9, 1.2, 1.5, 1.25, 2, 3])
# source_y_list = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# source_l_list = np.array([0.4, -0.05, -0.05, 0.2 , -0.05, -0.05, -0.1, -0.1, -0.05, -0.15])

# u_inf = 0.5
# v_inf = 0

# N = 1000

# x_bounds = np.linspace(-1, 4, N)
# y_bounds = np.linspace(-2, 2, N)

# X, Y = np.meshgrid(x_bounds, y_bounds)


# airfoil_geometry = pd.DataFrame()

# x, zup = NACAThicknessEquation(0, 0, 20, 1000)

# airfoil_geometry["NACA 0020 X"], airfoil_geometry["NACA 0020"] = x, zup



# airfoil_geometry *= 3
# stream2 = CreateStreamPlot(source_x_list, source_y_list, source_l_list, u_inf, v_inf, X, Y, N)

# # plt.xlim([-0.5, 3.5])
# plt.ylim([-2, 2.5])

# x *= 3
# zup *= 3
# ind = np.where(x < 3*0.75)
# x = x[ind]
# z = zup[ind]

# ind = np.where(z > 0)
# x = x[ind]
# z = z[ind]

# streamlinedat = stream2.collections[3].get_paths()[0].vertices
# streamlinedat = streamlinedat[np.where(streamlinedat[:, 1] > 0.0001)]
# streamlinedat = streamlinedat[np.where(streamlinedat[:, 0] > 0)]
# streamlinedat = streamlinedat[np.where(streamlinedat[:, 0] < 3*0.75)]

# a1 = np.trapz(z, x)
# a2 = np.trapz(streamlinedat[:, 1], streamlinedat[:, 0])

# arbitrary_error_metric = (a1 - a2)**2
# string = f'Arbitrary error metric value = {arbitrary_error_metric}'



# plothus(stream2, airfoil_geometry["NACA 0020 X"], airfoil_geometry["NACA 0020"],
#         datalabel='NACA 0020 Geometry')
# plt.axis('equal')

#%%###########################

Nlist = np.array([5, 11, 101])
u_inf = 0.5
v_inf = 0

N = 2000

x_bounds = np.linspace(-1, 1, N)
y_bounds = np.linspace(-0.5, 1.5, N)

X, Y = np.meshgrid(x_bounds, y_bounds)

graphlist = np.empty((len(Nlist)), dtype=object)
i = 0

for NN in Nlist:
    
    ypos = np.linspace(0, 1, NN)
    xpos = np.zeros((NN))
    x_other = 0.25 * np.ones((NN))
    l = (5/NN) * np.ones(NN)
    
    source_y_list = np.concatenate((ypos, ypos))
    source_x_list = np.concatenate((xpos, x_other))
    source_l_list = np.concatenate((l, -l))

    graphlist[i] = CreateStreamPlot(source_x_list, source_y_list, source_l_list, u_inf, v_inf, X, Y, N)
    plt.axis('equal')
    plt.ylim([-0.5, 1.5])
    i += 1

# plot "wall"
boxx = np.array([0.4, 0.4, -0.15, -0.15, 0.4])
boxy = np.array([-0.2, 1.2, 1.2, -0.2, -0.2])
graphlist[0].plot(boxx, boxy, linewidth = 3, linestyle='--', label='"Wall"')
    

boxx = np.array([0.41, 0.41, -0.18, -0.18 , 0.41])
boxy = np.array([-0.2, 1.2, 1.2, -0.2, -0.2])
graphlist[1].plot(boxx, boxy, linewidth = 3, linestyle='--', label='"Wall"')

boxx = np.array([0.5, 0.5, -0.25, -0.25, 0.5])
boxy = np.array([-0.2, 1.2, 1.2, -0.2, -0.2])
graphlist[2].plot(boxx, boxy, linewidth = 3, linestyle='--', label='"Wall"')
    

#%%###########################

# 4.1

xcp = lambda cm, cl: (-cm/cl + 0.25)

FourOneDat = pd.read_csv("Data/Prob4Dat.csv")

FourOneDat["X_cp"] = xcp(FourOneDat["C_m"], FourOneDat["C_l"])

fig, fourplot = plt.subplots()
plothusly(fourplot, FourOneDat["alpha"], FourOneDat["X_cp"],
          xtitle=r'Angle of attack $\alpha$ in degrees',
          ytitle=r'$C_P$ position $x_{C_{P}}$',
          title=r'$x_{C_{P}}$ as a function of $\alpha$',
          datalabel=r'$x_{C_{P}}$')

#%%###########################

# 4.2

reynolds = lambda rho, V, L, T: (rho*V*L/(T**0.5))
Ma = lambda V, T: (V/(T**0.5))

T = 199
fig, rocketplot = plt.subplots()
plothusly(rocketplot, rocketx, rockety, xtitle='x', ytitle='u', datalabel='', title='Problem 1b: Nozzle Flow')

rho = 1.23
V = 141
L = 1

re1 = reynolds(rho, V, L, T)
ma1 = Ma(V, T)

T = 400
rho = 1.739
V = 200
L = 2

re2 = reynolds(rho, V, L, T)
ma2 = Ma(V, T)

string1 = f'Re_1 = {re1:.2f}, Re_2 = {re2:.2f}'
string2 = f'Ma_1 = {ma1:.2f}, Ma_2 = {ma2:.2f}'

print(string1)
print(string2)

#%%###########################

# 4.3


#%%###########################

# 4.4


