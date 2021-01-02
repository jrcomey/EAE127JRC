#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 12:32:38 2020

@author: jack
"""

# Imports 

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import copy
import pandas as pd
import ViscousInviscidInteraction as vivi
import pyxfoil as pyx
import mses
plt.style.use('default')
plt.style.use("seaborn-bright")

params={#FONT SIZES
#     'axes.labelsize':30,#Axis Labels
    'axes.titlesize':30,#Title
    # 'font.size':20,#Textbox
#     'xtick.labelsize':22,#Axis tick labels
#     'ytick.labelsize':22,#Axis tick labels
#     'legend.fontsize':24,#Legend font size
#     'font.family':'sans-serif',
#     'font.fantasy':'xkcd',
#     'font.sans-serif':'Helvetica',
#     'font.monospace':'Courier',
#     #AXIS PROPERTIES
#     'axes.titlepad':2*6.0,#title spacing from axis
#     'axes.grid':True,#grid on plot
    'figure.figsize':(12,12),#square plots
#     'savefig.bbox':'tight',#reduce whitespace in saved figures#LEGEND PROPERTIES
#     'legend.framealpha':0.5,
#     'legend.fancybox':True,
#     'legend.frameon':True,
#     'legend.numpoints':1,
#     'legend.scatterpoints':1,
#     'legend.borderpad':0.1,
#     'legend.borderaxespad':0.1,
#     'legend.handletextpad':0.2,
#     'legend.handlelength':1.0,
    'legend.labelspacing':0,}
mpl.rcParams.update(params)

#%###########################

# Objects

#%###########################

# Functions

def FindReynolds(rho, V, L, mu):
    Re = rho * V * L / mu
    return Re

def CreateCamberLine(df):
    
    
    x = np.linspace(0, 1, 1000)
    up, lo = mses.MsesInterp(x, df.x, df.z)


    avg = up + lo
    avg *= 0.5
    
    return x, avg

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
    ax.grid(True)
    ax.legend(loc='best')
    return out


def plothus(ax, x, y, *, datalabel='', linestyle = '-',
            marker = ''):
    """
    A little function to make graphing less of a pain

    Adds a new line to a blank figure and labels it
    """
    out = ax.plot(x, y, zorder=1, label=datalabel, linestyle = linestyle,
                  marker = marker)
    ax.legend(loc='best')

    return out

#%###########################

# Main

# Problem 1.1

# Initialize nonvariable properties

mu = 3.737E-7  # slug/ft/s
L = 0.5  # ft

# Unit conversion:
kts2fps = lambda v: (1.68781 * v)

# Condition 1

rho_1 = 0.00237717  # slug/ft**3

Reynolds_1 = FindReynolds(rho_1, kts2fps(12), L, mu)
print(f"Reynold's # at condition 1: {Reynolds_1:.2e}")


# Condition 2

rho_2 = 0.000825628  # slug/ft**3

Reynolds_2 = FindReynolds(rho_2, kts2fps(575), L, mu)
print(f"Reynold's # at condition 2: {Reynolds_2:.2e}")

#%%###########################

# Problem 1.2

# Laminar boundary layer

laminar_boundary_u_nondim = lambda y_delta: ( 2 * y_delta - (y_delta**2))

# Turbulent Boundary Layer
turbulent_boundary_u_nondim = lambda y_delta: (y_delta**(1/7))

# Data calculation

y_non_dim = np.linspace(0, 1, 1000)

laminar_u_non_dim = laminar_boundary_u_nondim(y_non_dim)
turbulent_u_non_dim = turbulent_boundary_u_nondim(y_non_dim)

fig, non_dim_boundary_layer = plt.subplots()

plothusly(non_dim_boundary_layer,
        laminar_u_non_dim,
        y_non_dim,
        title="Non-Dimensional Boundary Layer Comparison",
        xtitle=r'$\frac{u}{u_e}$',
        ytitle=r"$\frac{y}{\delta}$",
        datalabel="Taxi")

plothus(non_dim_boundary_layer,
        turbulent_u_non_dim,
        y_non_dim,
        datalabel="Cruise")


# Shade in boundary layer
vline = y_non_dim*0
plothus(non_dim_boundary_layer, laminar_u_non_dim, vline, linestyle='')
plt.fill_betweenx(y_non_dim, vline, laminar_u_non_dim, facecolor='b', alpha=0.1)
plt.axis('equal')

# Make Arrows
arrowwidth, arrowlength = 0.02, 0.02

for i in range(0, len(y_non_dim), 50):
    if abs(laminar_u_non_dim[i]) < arrowlength:
        plt.plot([0, laminar_u_non_dim[i]], [y_non_dim[i], y_non_dim[i]], color='b')
    else:
        plt.arrow(0, y_non_dim[i], laminar_u_non_dim[i]-arrowlength, 0, head_width=arrowwidth,
                  head_length=arrowlength, color='b', linewidth=2, alpha=0.2)

plothus(non_dim_boundary_layer, turbulent_u_non_dim, vline, linestyle='')
plt.fill_betweenx(y_non_dim, vline, turbulent_u_non_dim, facecolor='g', alpha=0.1)
plt.axis('equal')

# Make Arrows
arrowwidth, arrowlength = 0.02, 0.02

for i in range(0, len(y_non_dim), 50):
    if abs(turbulent_u_non_dim[i]) < arrowlength:
        plt.plot([0, turbulent_u_non_dim[i]], [y_non_dim[i], y_non_dim[i]], color='g')
    else:
        plt.arrow(0, y_non_dim[i], turbulent_u_non_dim[i]-arrowlength, 0, head_width=arrowwidth,
                  head_length=arrowlength, color='g', linewidth=2, alpha=0.2)
        


# Laminar Dimensional

delta_x_lam = lambda Re, x: 5.0 * x * (Re**(-1/2))
laminar_y = y_non_dim * delta_x_lam(Reynolds_1, L)
laminar_u = kts2fps(575) * laminar_boundary_u_nondim(laminar_y)
# Turbulent Dimensional

turbulent_u = turbulent_u_non_dim * kts2fps(575)
delta_x_tur = lambda Re, x: 0.16 * x * (Re**(-1/7))
turbulent_y = y_non_dim * delta_x_tur(Reynolds_2, L)


fig, boundary_layer_plot = plt.subplots()

plothusly(boundary_layer_plot,
          laminar_u,
          laminar_y,
          title="Dimensional Boundary Layer Comparison",
          ytitle=r"$y$ [ft]",
          xtitle=r"$u$ [ft/s]",
          datalabel="Taxi")

plothus(boundary_layer_plot,
        turbulent_u,
        turbulent_y,
        datalabel="Cruise")


# Shade in boundary layer
vline = laminar_y*0
plothus(boundary_layer_plot, laminar_u, vline, linestyle='')
plt.fill_betweenx(laminar_y, vline, laminar_u, facecolor='b', alpha=0.1)

vline = turbulent_y*0
plothus(boundary_layer_plot, turbulent_u, vline, linestyle='')
plt.fill_betweenx(turbulent_y, vline, turbulent_u, facecolor='g', alpha=0.1)

#%%###########################

# Problem 1.3 

Cflam = lambda Re: 1.328 * (Re**(-1/2))

C_f_laminar = 2 * Cflam(Reynolds_1)

string = f"C_f for condition 1: {C_f_laminar:.4e}"
print(string)


Cftur = lambda Re: 0.074 * (Re**(-1/5))

C_f_turbulent = 2*Cftur(Reynolds_2)
string = f"C_f for condition 2: {C_f_turbulent:.4e}"
print(string)

cf_lam_long = 2 * Cflam(FindReynolds(rho_1, kts2fps(12), 2*L, mu))
cf_tur_long = 2 * Cftur(FindReynolds(rho_2, kts2fps(575), 2*L, mu))

drag_lam_long = 0.5 * rho_1 * kts2fps(12)**2 * cf_lam_long
drag_tur_long = 0.5 * rho_2 * kts2fps(575)**2 * cf_tur_long

string = f"Drag force for double-length antenna at condition 1 is : {drag_lam_long:.5f} lbf"
print(string)

string = f"Drag force for double-length antenna at condition 2 is : {drag_tur_long:.5f} lbf"
print(string)
#%%###########################

# Problem 2.1

airfoil_name = "naca23012"
alpha = 0
currentiter = 0
V_inf = 1  # m/s
rho = 1.225  # kg/m**3
mu = 1.789E-5
itermax = 4


df_mses = pyx.ReadXfoilAirfoilGeom('Data/naca23012/naca23012.dat')

fig, viviplot = plt.subplots(figsize=(12,3))
fig, viviplot2 = plt.subplots(figsize=(12,3))


viviplot.axis('equal')

plothusly(viviplot,
          df_mses["x"],
          df_mses["z"],
          datalabel=r"0$^{th}$",
          xtitle=r"$\frac{x}{c}$",
          ytitle=r"$\frac{z}{c}$",
          title="VIvI Iteration Comparison")



plothusly(viviplot2,
          df_mses["x"],
          df_mses["z"],
          datalabel=r"",
          xtitle=r"$\frac{x}{c}$",
          ytitle=r"$\frac{z}{c}$",
          title="VIvI Iteration Comparison")

x_avg, z_avg = CreateCamberLine(df_mses)

plothus(viviplot2,
        x_avg,
        z_avg,
        datalabel=fr"Average Camber for Iteration {currentiter}")

plt.axis("equal")

fig, camberplot = plt.subplots()

plothusly(camberplot,
          x_avg,
          z_avg,
          datalabel=fr"Average Camber for Iteration {currentiter}")

for currentiter in range(itermax):
    
    theta_up, ue_up, theta_lo, ue_lo = vivi.VIvI(airfoil_name,
                                                 alpha,
                                                 currentiter,
                                                 V_inf,
                                                 mu,
                                                 rho)
    
    
    if currentiter is not 0:
        df_disp = pyx.ReadXfoilAirfoilGeom(f"Data/naca23012/naca23012_{currentiter}.dat")
    
        plothus(viviplot,
                df_disp["x"],
                df_disp["z"],
                datalabel=rf"{currentiter}",
                linestyle='-')
        
        plothus(viviplot2,
                df_disp["x"],
                df_disp["z"],
                datalabel=rf"",
                linestyle='--')
        
        x, z = CreateCamberLine(df_disp)
        
        plothus(viviplot2,
                x, z,
                datalabel=f"Average Camber for Iteration {currentiter}")
        
        plothus(camberplot,
                x, z,
                datalabel=f"Average Camber for Iteration {currentiter}")
    
    
    
#%%###########################

# Problem 2.2

pyx.GetPolar(foil="Data/naca23012_3/naca23012_3.dat", naca=False, alfs=alpha, Re=0)

third_iter_dat = pyx.ReadXfoilPolar("Data/naca23012_3/naca23012_3_polar_Re0.00e+00a0.00.dat")


print(third_iter_dat.to_markdown())

theta_lo = theta_lo.to_numpy()[2:len(theta_lo)]
theta_up = theta_up.to_numpy()[2:len(theta_up)]
ue_up = ue_up.to_numpy()[2:len(ue_up)]
ue_lo = ue_lo.to_numpy()[2:len(ue_lo)]

# Find tau at every point
tau_find = lambda mu, ue, theta: 0.664 * 0.332 * mu * ue / theta



tau_lo = tau_find(mu, ue_lo, theta_lo)
tau_up = tau_find(mu, ue_up, theta_up)

cf = lambda tau: tau * (0.5 * rho * V_inf**2)**-1

cf_lo = cf(tau_lo)
cf_up = cf(tau_up)

x = df_disp.x.to_numpy()
x = np.flip(x[2:len(tau_lo)+2])


Cf_lo = np.trapz(cf_lo, x)
Cf_up = np.trapz(cf_up, x)

Cf = Cf_lo + Cf_up


string = f'Friction Coefficent for 3rd iteration = {Cf}'
print(string)

Re_3 = FindReynolds(rho, V_inf, 1, mu)

string = f"Reynold's number is {Re_3:.2e}"
print(string)

pyx.GetPolar(foil='23012', naca=True, alfs=alpha, Re=Re_3)
pyx.GetPolar(foil='23012', naca=True, alfs=alpha, Re=0)
visc = pyx.ReadXfoilPolar("Data/naca23012/naca23012_polar_Re6.85e+04a0.00.dat")
invisc = pyx.ReadXfoilPolar("Data/naca23012/naca23012_polar_Re0.00e+00a0.00.dat")


#%%###########################

# Problem 2.3

