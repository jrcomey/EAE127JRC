#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 16:15:17 2020

@author: jack
"""

# Imports 

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import copy
import VortexPanelMethod as vpm
import pandas as pd
import pyxfoil as pyx
plt.style.use('classic')


params={#FONT SIZES
    'axes.labelsize':30,#Axis Labels
    'axes.titlesize':30,#Title
    'font.size':28,#Textbox
    'xtick.labelsize':22,#Axis tick labels
    'ytick.labelsize':22,#Axis tick labels
    'legend.fontsize':24,#Legend font size
    'font.family':'sans-serif',
    'font.fantasy':'xkcd',
    'font.sans-serif':'Helvetica',
    'font.monospace':'Courier',
    #AXIS PROPERTIES
    'axes.titlepad':2*6.0,#title spacing from axis
    'axes.grid':True,#grid on plot
    'figure.figsize':(12,12),#square plots
    'savefig.bbox':'tight',#reduce whitespace in saved figures#LEGEND PROPERTIES
    'legend.framealpha':0.5,
    'legend.fancybox':True,
    'legend.frameon':True,
    'legend.numpoints':1,
    'legend.scatterpoints':1,
    'legend.borderpad':0.1,
    'legend.borderaxespad':0.1,
    'legend.handletextpad':0.2,
    'legend.handlelength':1.0,
    'legend.labelspacing':0,}
mpl.rcParams.update(params)


#%###########################

# Objects

#%###########################

# Functions


def NACAThicknessEquationMOD(N, A, CA, num_points, *, use_other_x_points=0):
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
    
    x_non_dim = np.concatenate((np.flip(x_non_dim), x_non_dim[1:]))
    z = np.concatenate((np.flip(zup), zdown[1:]))
    return x_non_dim, z

def CalculateCirculation(xpath, ypath, panel_list, V_inf, alpha):
    
    n = len(xpath)
    
    xc = np.zeros(n-1)
    yc = np.zeros(n-1)
    s = np.zeros(n-1)
    path_angle = np.zeros(n-1)
    

    for i in range(n-1):
        xc[i] = (xpath[i] + xpath[i+1]) / 2
        yc[i] = (ypath[i] + ypath[i+1]) / 2
        s[i] = ((xpath[i+1] - xpath[i])**2 + (ypath[i+1] - ypath[i])**2)**0.5
        path_angle[i] = np.arctan2(ypath[i+1] - ypath[i], xpath[i+1] - xpath[i])

    u, v = vpm.GetVelocity(xpath, ypath, panel_list, V_inf, alpha)
    v_angle = np.arctan2(v, u)
    V = (u**2 + v**2)**0.5
    Gamma = 0
    
    for i in range(n-1):
        Gamma += - V[i] * np.cos(v_angle[i] - path_angle[i]) * s[i]
    
    return Gamma

def KuttaJoukowskiLift(Gamma, rho, V_inf):
    
    Lprime = rho * V_inf * Gamma
    return Lprime

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
    plt.grid(True)
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

def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return rad * 180 / np.pi
#%%###########################

# Main

# Generate Airfoil Geometry

nacaX, nacaZ = NACAThicknessEquationMOD(0, 0, 12, 40)
nacaX, nacaZ = np.flip(nacaX), np.flip(nacaZ)

seligX, seligZ = vpm.ReadXfoilGeometry("AirfoilData/s1223.dat")

# Create Freestream Properties

V_fs = 1.5  # m-s**-1
rho = 1.2  # kg-m**-3
alphalist = deg2rad(np.array([0, 8]))
num_panels = 60

panel_enforced_index = 30  # ndim
missing_panel_index = panel_enforced_index - 1

# GENERALIZED LOOP

xdat, ydat = seligX, seligZ
airfoilname = "Selig 1223"
for alpha in alphalist:
    
    panel_list = vpm.MakePanels(xdat, ydat, num_panels, 'constant')
    panel_list = vpm.SolveVorticity(panel_list, missing_panel_index,
                                    V_fs, alpha)

    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-1.5, 1.5, 100)
    X, Y = np.meshgrid(x, y)
    fig, flowplot = plt.subplots()
    u, v = vpm.GetVelocity(X, Y, panel_list, V_fs, alpha)
    V = (u**2 + v**2)**0.5
    fig, flowplot = plt.subplots()
    plt.streamplot(X, Y, u, v, density=2, linewidth=1)
    plt.contourf(X, Y, V, cmap='hot', alpha=0.5)
    cbar = plt.colorbar()
    cbar.set_label("Fluid Velocity (m/s)")
    plt.axis('equal')
    plt.plot(xdat, ydat, color='k', linewidth=2)
    plt.title(fr"Airflow plot for {airfoilname}, $\alpha$ = {rad2deg(alpha)}")
#%%  

xdat, ydat = nacaX, nacaZ
airfoilname = "NACA 0012"
for alpha in alphalist:
    
    panel_list = vpm.MakePanels(xdat, ydat, num_panels, 'constant')    
    panel_list = vpm.SolveVorticity(panel_list, missing_panel_index,
                                    V_fs, alpha)

    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-1.5, 1.5, 100)
    X, Y = np.meshgrid(x, y)
    u, v = vpm.GetVelocity(X, Y, panel_list, V_fs, alpha)
    V = (u**2 + v**2)**0.5
    fig, flowplot = plt.subplots()
    plt.streamplot(X, Y, u, v, density=5, linewidth=0.5)
    plt.contourf(X, Y, V, cmap='hot', alpha=0.5)
    cbar = plt.colorbar()
    cbar.set_label("Fluid Velocity (m/s)")
    plt.axis('equal')
    plt.plot(xdat, ydat, color='k', linewidth=2)
    plt.title(f"Airflow plot for {airfoilname}, alpha = {rad2deg(alpha)}")

#%%###########################

# Create two seperate integration paths

# Rectangular integration path

x_corners = [-0.5, 1.5, 1.5, -0.5]
y_corners = [-0.5, -0.5, 0.5, 0.5]

n = 50
integration_path_1_x = np.concatenate((np.linspace(x_corners[0], x_corners[1], n),
                                       np.linspace(x_corners[1], x_corners[2], n),
                                       np.linspace(x_corners[2], x_corners[3], n),
                                       np.linspace(x_corners[3], x_corners[0], n),))

integration_path_1_y = np.concatenate((np.linspace(y_corners[0], y_corners[1], n),
                                       np.linspace(y_corners[1], y_corners[2], n),
                                       np.linspace(y_corners[2], y_corners[3], n),
                                       np.linspace(y_corners[3], y_corners[0], n),))

# Circular Path 

integration_path_2_x = 0.75 * np.cos(np.linspace(0, 2 * np.pi, n*4)) + 0.5
integration_path_2_y = np.sin(np.linspace(0, 2 * np.pi, n*4)) / 4

# PLOTS 
fig, flowplot = plt.subplots()
plt.streamplot(X, Y, u, v, density=5, linewidth=0.5, cmap='coolwarm')
plt.axis('equal')
plt.plot(xdat, ydat, color='k', linewidth=2)
plt.plot(integration_path_1_x, integration_path_1_y, linewidth=4, color='k', linestyle='--')
plt.title(f"Airflow plot for {airfoilname}, alpha = {rad2deg(alpha)}")

fig, flowplot = plt.subplots()
plt.streamplot(X, Y, u, v, density=5, linewidth=0.5, cmap='coolwarm')
plt.axis('equal')
plt.plot(xdat, ydat, color='k', linewidth=2)
plt.plot(integration_path_2_x, integration_path_2_y, linewidth=4, color='k', linestyle='--')
plt.title(f"Airflow plot for {airfoilname}, alpha = {rad2deg(alpha)}")

# Calculate Circulations

Circulation = CalculateCirculation(integration_path_1_x, integration_path_1_y, panel_list, V_fs, alpha)
print(f'Circulation is {Circulation} for {n*4} items on a rectangular path')

rectangle_lift = KuttaJoukowskiLift(Circulation, rho, V_fs)

Circulation = CalculateCirculation(integration_path_2_x, integration_path_2_y, panel_list, V_fs, alpha)
print(f'Circulation is {Circulation} for {n*4} items on a circular path')

elliptical_lift = KuttaJoukowskiLift(Circulation, rho, V_fs)


# Find Cl for comparison

C_l_rectangle = rectangle_lift / (0.5 * rho * V_fs**2 * 1)
C_l_elliptical = elliptical_lift / (0.5 * rho * V_fs**2 * 1)

#%%###########################

N = 26

alphalist = np.deg2rad(np.linspace(0, 25, N))

KuttaJoukowskiC_l = np.zeros(N)
ThinAirfoilC_l = 2 * np.pi * alphalist

xfoildat = pd.read_csv("AirfoilData/NACA0012Xfoildata.csv")
for i, alpha in enumerate(alphalist):
    panel_list = vpm.MakePanels(xdat, ydat, num_panels, 'constant')    

    panel_list = vpm.SolveVorticity(panel_list, missing_panel_index,
                                    V_fs, alpha)
    Gamma = CalculateCirculation(integration_path_2_x, integration_path_2_y, 
                                 panel_list, V_fs, alpha)
    elliptical_lift = KuttaJoukowskiLift(Gamma, rho, V_fs)
    KuttaJoukowskiC_l[i] = elliptical_lift / (0.5 * rho * V_fs**2 * 1)

fig, lift_comparison = plt.subplots()
plothusly(lift_comparison,
          np.rad2deg(alphalist),
          KuttaJoukowskiC_l,
          title=fr'$C_l$ Comparison for varying mathods for {airfoilname} \n ',
          xtitle=fr'$\alpha$',
          ytitle=fr'$C_l$',
          datalabel="Vortex panel / Kutta-Joukowski Theorem", 
          marker='x')
plothus(lift_comparison,
        np.rad2deg(alphalist),
        ThinAirfoilC_l,
        datalabel='Thin Airfoil Theory',
        marker='p')
plothus(lift_comparison,
        xfoildat["Alpha"],
        xfoildat["C_l"],
        datalabel='Viscous Xfoil, Re = 5E6')


#%%###########################

alpha = np.deg2rad(45)

panel_list = vpm.MakePanels(xdat, ydat, num_panels, 'constant')    
panel_list = vpm.SolveVorticity(panel_list, missing_panel_index,
                                    V_fs, alpha)

x = np.linspace(-0.5, 1.5, 20)
y = np.linspace(-1.5, 1.5, 20)
X, Y = np.meshgrid(x, y)
u, v = vpm.GetVelocity(X, Y, panel_list, V_fs, alpha)
V = (u**2 + v**2)**0.5
fig, flowplot = plt.subplots()
plt.streamplot(X, Y, u, v, density=5, linewidth=0.5)
plt.contourf(X, Y, V, cmap='viridis', alpha=0.5, levels=np.linspace(0, 7, 8))
cbar = plt.colorbar()
cbar.set_label("Fluid Velocity (m/s)")
plt.axis('equal')
plt.plot(xdat, ydat, color='k', linewidth=2)
plt.title(f"Airflow plot for {airfoilname}, alpha = {rad2deg(alpha)}")


#%%###########################

# 4.1

mph2fps = lambda v: v * 1.46667

W = 1669  # lbf
b = 36  # ft
rho = 0.00237717  # slug/ft**3
V = mph2fps(140)  # ft/s


Gamma = W / (rho * V * b)

R = np.linspace(1, 20, 2001)
Omega = Gamma / (2 * np.pi * R)**2

fig, omegaplot = plt.subplots()

plothusly(omegaplot,
          R,
          Omega,
          xtitle=r"Rotor Radius $R$ (ft)",
          ytitle=r'Rotor speed $\Omega$',
          title="Rotorcraft Speed Requirements",
          datalabel="Cessna 172R")

#%%###########################

# 4.2

prob42dat = pd.read_csv("AirfoilData/VariedNACAdat.csv")

# Standard Atmosphere Properties

column_names = ["NACA 2412 Lift", "NACA 2412 Drag", 
                "NACA 4412 Lift", "NACA 4412 Drag",
                "NACA 23012 Lift", "NACA 23012 Drag"]

Liftdat = pd.DataFrame(columns=column_names)

rho = 0


Re = 9E6 
chord = 7 # ft
rho = 20.48E-4 # slug/ft**3

mu = 3.637E-7 # slug  / (ft-s) FROM ENGINEERING TOOLBOX


# Formula for v_inf
# v_inf = Re * mu / (rho * chord)


const2force = lambda C, Re: (0.5 * rho * C * chord * (Re * mu / (rho * chord))**2)

Liftdat["NACA 2412 Lift"] = const2force(prob42dat["NACA 2412 C_l"],
                                        prob42dat["Reynolds"])

Liftdat["NACA 2412 Drag"] = const2force(prob42dat["NACA 2412 C_d"],
                                        prob42dat["Reynolds"])

Liftdat["NACA 4412 Lift"] = const2force(prob42dat["NACA 4412 C_l"],
                                        prob42dat["Reynolds"])

Liftdat["NACA 4412 Drag"] = const2force(prob42dat["NACA 4412 C_d"],
                                        prob42dat["Reynolds"])

Liftdat["NACA 23012 Lift"] = const2force(prob42dat["NACA 23012 C_l"],
                                        prob42dat["Reynolds"])

Liftdat["NACA 23012 Drag"] = const2force(prob42dat["NACA 23012 C_d"],
                                        prob42dat["Reynolds"])

print(Liftdat.to_markdown())

#%%###########################

# 4.3 

airfoildatabase = pd.DataFrame()

alphalist = np.linspace(0, 25, 26)

# pyx.GetPolar(foil='2412', naca=True, alfs=alphalist, Re=9E6)
naca2412 = pyx.ReadXfoilPolar("Data/naca2412/naca2412_polar_Re9.00e+06a0.0-25.0.dat")

# pyx.GetPolar(foil='4412', naca=True, alfs=alphalist, Re=9E6)
naca4412 = pyx.ReadXfoilPolar("Data/naca4412/naca4412_polar_Re9.00e+06a0.0-25.0.dat")

# pyx.GetPolar(foil='23012', naca=True, alfs=alphalist, Re=9E6)
naca23012 = pyx.ReadXfoilPolar("Data/naca23012/naca23012_polar_Re9.00e+06a0.0-25.0.dat")

findLD = lambda df: (df["Cl"] / df["Cd"])

naca2412["L/D"] = findLD(naca2412)
naca4412["L/D"] = findLD(naca4412)
naca23012["L/D"] = findLD(naca23012)

fig, LDplot = plt.subplots()

plothusly(LDplot, 
          naca2412["alpha"],
          naca2412["L/D"],
          title="L/D Ratio comparison for Multiple Airfoils",
          xtitle=r"$\alpha$",
          ytitle=r"$\frac{L}{D}$",
          datalabel="NACA 2412",
          marker='p')

plothus(LDplot,
        naca4412["alpha"],
        naca4412["L/D"],
        datalabel="NACA 4412",
        marker='o')

plothus(LDplot,
        naca23012["alpha"],
        naca23012["L/D"],
        datalabel="NACA 23012",
        marker='d')
#%%###########################

# 4.4 

kmhr2mps = lambda v: v * 0.277778
slugft3_2_kgm3 = lambda slug: slug * 515.379


e = 0.75
AR = 9
S = 18.3  # m/s**20.277778
W = 9E3  # N

Clfind = lambda V, rho: W / (0.5 * rho * V**2 * S)
Cdindfind = lambda Cl: Cl**2 / (np.pi * e * AR)
Dfind = lambda Cd, V, rho: 0.5 * rho * V**2 * S * Cd



# Cruise

V = kmhr2mps(245)  # m/s
rho = slugft3_2_kgm3(0.000825628)  # kg/m**3



cruiseCL = Clfind(V, rho)
cruiseCD = Cdindfind(cruiseCL)
cruiseD = Dfind(cruiseCD, V, rho)

# Landing

V = kmhr2mps(85)  # m/s
rho = 1.22500  # m/s

landCL = Clfind(V, rho)
landCD = Cdindfind(landCL)
landD = Dfind(landCD, V, rho)


column_names = ["Cruise Drag Force (N)"]
df = pd.DataFrame(np.array([cruiseD]), columns = column_names)

df["Landing Drag Force (N)"] = landD

print(df.to_markdown())

#%%###########################

x, y = NACAThicknessEquationMOD(2, 4, 12, 100)

fig, geoplot = plt.subplots()

plothusly(geoplot, x, y,
          xtitle=r"$\frac{x}{c}$",
          ytitle=r'$\frac{z}{c}$',
          title="NACA 2412 Geometry",
          datalabel="NACA 2412")
plt.fill(x, y,)
plt.axis('equal')