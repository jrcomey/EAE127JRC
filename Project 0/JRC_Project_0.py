#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:12:03 2020

@author: jack
"""

# Imports 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("classic")
#%############################

# Objects

#%############################

# Functions

def plothusly(ax, x, y, *, xtitle='', ytitle='',
              datalabel='', title='', linestyle = '-',
              marker = ''):
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
    return out


def plothus(ax, x, y, *, datalabel='', linestyle = '-',
              marker = ''):
    """
    A little function to make graphing less of a pain

    Adds a new line to a blank figure and labels it
    """
    out = ax.plot(x, y, zorder=1, label=datalabel, linestyle = linestyle,
                  marker = marker)
    return out


#%%###########################

# Problem 1 a
ynon = np.linspace(0, 1, 1000)
unon = ynon**(1/7)

fig, prob1plot = plt.subplots()
plothusly(prob1plot, unon, ynon, xtitle='$u/u_e$', ytitle='y/$\delta$', datalabel='Boundary layer', title='Boundary Layer Velocity Profile', linestyle='solid')
plt.legend(loc='best')
plt.grid()



#%%###########################

# Problem 1b

Re = 1E8  # ndim
pos = 300  # ft

delta_star_non_dim = np.trapz(1 - unon, ynon)
delta_formula = lambda x : (0.16*x) / ((Re**(1/7)))
delta_point = delta_formula(pos)
delta_star = delta_star_non_dim * delta_point
print(delta_point*12)
print(delta_star*12)


#%%###########################

# Problem 2a
x = 300
Re_x = 1E8

delx = 0.16*x * (Re_x)**(-1/7)

s6063 = np.loadtxt('Data/s6063.dat')
s8036 = np.loadtxt('Data/s8036.dat')
tempest = np.loadtxt('Data/tempest1.dat')
s6063df = pd.DataFrame(s6063)
s8036df = pd.DataFrame(s8036)
tempestdf = pd.DataFrame(tempest)


fig, airfoilplot = plt.subplots()
plothusly(airfoilplot, s6063df[0], s6063df[1], marker='', linestyle='-', datalabel='Selig 6063')
plothus(airfoilplot, s8036df[0], s8036df[1], marker='', linestyle='--', datalabel='Selig 8036')
plothus(airfoilplot, tempestdf[0], tempestdf[1], marker='', linestyle='-.', datalabel='Hawker Tempest 37.5% Semi-span')
plt.legend(loc='best')
plt.axis('equal')

#%%###########################

# Problem 2b


c = 9.5  # ft, chord length
naca_2412_data = np.loadtxt('Data/naca2412_geom.dat', unpack=True, skiprows=1)
naca_2412_data *= c  # Redimensionalize
naca_2412_df = pd.DataFrame(naca_2412_data.transpose())
area = -np.trapz(naca_2412_df[1], naca_2412_df[0])
string = f"""Cross-sectional area of NACA 2412 airfoil
with chord length {c} is {area} feet"""
print(string)

#%%###########################

# Problem 3a 

pressure_distribution = pd.read_csv('Data/naca2412_SurfPress_a6.csv')
fig, presdistplot = plt.subplots()
plothusly(presdistplot, pressure_distribution["x"], -pressure_distribution["Cpl"], xtitle = 'x/c', ytitle='C$_P$', title="Surface Pressure Distribution", datalabel='Lower', linestyle='-') 
plothus(presdistplot, pressure_distribution['x'], -pressure_distribution['Cpu'], datalabel='Upper', linestyle='-')
plt.grid()
plt.legend(loc='best')


#%%###########################

# Problem 3b


pressure_gradient = np.zeros((len(pressure_distribution), 1))
for i in range(len(pressure_distribution)-1):
    pressure_gradient[i] = ((pressure_distribution["Cpl"][i+1] 
                             - pressure_distribution["Cpl"][i])
                            / (pressure_distribution['x'][i+1]
                               - pressure_distribution['x'][i]))
pressure_distribution['Gradl'] = pressure_gradient
pressure_gradient *= 0

for i in range(len(pressure_distribution)-1):
    pressure_gradient[i] = ((pressure_distribution["Cpu"][i+1] 
                              - pressure_distribution["Cpu"][i])
                            / (pressure_distribution['x'][i+1]
                                - pressure_distribution['x'][i]))
pressure_distribution['Gradu'] = pressure_gradient

fig, presgradplot = plt.subplots()

plothusly(presgradplot, pressure_distribution["x"],
          pressure_distribution["Gradl"], xtitle = 'x/c',
          ytitle='C$_P$', title="Surface Pressure Gradient",
          datalabel='Lower', linestyle='-') 

plothus(presgradplot, pressure_distribution['x'],
        pressure_distribution['Gradu'], datalabel='Upper',
        linestyle='-')
plt.xlim([0, 1])
plt.ylim([-10, 10])
plt.grid()
plt.legend(loc='best')

#%%###########################

# Problem 4a

naca_2412_lift_curve = pd.read_excel("Data/naca2412_LiftCurve.xlsx")
fig, liftplot = plt.subplots()
plothusly(liftplot, naca_2412_lift_curve['alpha'], naca_2412_lift_curve["Cl"],
          xtitle=r'Angle of Attack $\alpha$', ytitle=r'C$_l$',
          title='NACA 2412 Lift Curve', marker='o')
plt.grid()

c_l_interpolated = np.interp(5.65, naca_2412_lift_curve["alpha"],
                             naca_2412_lift_curve["Cl"])
string = rf"Interpolated C_l at alpha = 5.65 deg is {c_l_interpolated}"
print(string)


#%%###########################

A = np.array([[1, 2, 3, 4],
              [3, 2, -2, 3],
              [0, 1, 1, 0],
              [2, 1, 1, -2]])
b = np.array([[12],
              [10],
              [-1],
              [-5]])
x = np.linalg.solve(A, b)
print(x)