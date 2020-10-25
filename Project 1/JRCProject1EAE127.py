#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 14:38:28 2020

@author: jack
"""

# Imports 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import *
plt.style.use("classic")

#%%###########################

class Airplane115():
    
    def __init__(self, wing_area, weight, rho):
        """
        Airplane used in problem 1.15

        Parameters
        ----------
        wing_area : Wing area in ft^2
        weight : Aircraft weight in lbs
        rho : Air density in slug

        Returns
        -------
        None.

        """
        self.wing_area = wing_area
        self.weight = weight
        self.rho = rho
        
    def getProperties(self, vel_inf):
        """
        Returns properties given current speed

        Parameters
        ----------
        vel_inf : Current velocity in ft/s

        Returns
        -------
        C_L : Lift Coefficient
        C_D : Drag Coefficient
        LD : TLift/Drag Ratio

        """
        
        # From steady, level flight condition:
        L = self.weight
        C_L = L * 2 / (self.rho * vel_inf**2 * self.wing_area)
        
        CDFunc = lambda CL: (0.025 + 0.054 * C_L**2)
        C_D = CDFunc(C_L)
        
        D = 0.5 * (self.rho * vel_inf**2 * self.wing_area * C_D)
        LD = L/D
        return C_L, C_D, LD

#%%###########################

# Functions

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
    p = 0.1 * A
    m = 0.01 * N
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

    return x_non_dim, zup, zdown, zcc


def MomentumDeficit(y_non_dim, u1, u2, rho):
    """
    Calculates momentum defecit from incoming and outgoing fluid velocity

    Parameters
    ----------
    y_non_dim : Non-dimensionalized y values
    u1 : Fluid entrance profile
    u2 : Fluid exit profile
    rho : Fluid density

    Returns
    -------
    Dprime : Sectional Drag D'

    """

    internal = u2 * (u1 - u2)
    Dprime = np.trapz(internal, y_non_dim)
    Dprime *= rho
    return Dprime

def CalculateCn(Cplx, Cply, Cpux, Cpuy, *, c=1):
    left = np.trapz(Cply, Cplx)
    right = np.trapz(Cpuy, Cpux)
    out = left - right
    out /= c
    return out

def CalculateCa(Cplx, Cply, Cpux, Cpuy, xc, ztu, ztl,  *, c=1):
    gradu = np.gradient(ztu, xc)
    gradl = np.gradient(ztl, xc)
    
    try:
        left = np.trapz(Cpuy*gradu, Cpux)
        right= np.trapz
    except:
        pass
    

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
    left = np.trapz(Cpuy, Cpux)
    right = np.trapz(Cply, Cplx)
    CmLE = left - right
    CmLE /= c**2
    return CmLE

def RotateC(C_n, C_a, alpha):
    C_l = C_n*np.cos(np.deg2rad(alpha)) - C_a * np.sin(np.deg2rad(alpha))
    C_d = C_n*np.sin(np.deg2rad(alpha)) + C_a * np.cos(np.deg2rad(alpha))

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

# Problem 1


x, upper, lower, chord = NACAThicknessEquation(0, 0, 18, 100)
column_names = ["NACA 0018 X", "NACA 0018 Upper"]
airfoil_df = pd.DataFrame(columns=column_names)

airfoil_df["NACA 0018 X"], airfoil_df["NACA 0018 Upper"]= x, upper
airfoil_df["NACA 0018 Lower"], airfoil_df["NACA 0018 Chord"] = lower, chord



fig, NACAsymplot = plt.subplots()
plothusly(NACAsymplot,
          airfoil_df["NACA 0018 X"],
          airfoil_df["NACA 0018 Upper"],
          xtitle=r'Non-dimensionalized x-position $\frac{x}{c}$',
          ytitle=r'Non-dimensionalized z-position $\frac{z}{c}$',
          datalabel="NACA 0018 Upper Surface",
          title="NACA 0018 Airfoil Plot")

plothus(NACAsymplot,
        airfoil_df["NACA 0018 X"],
        airfoil_df["NACA 0018 Lower"],
        datalabel='NACA 0018 Lower Surface',
        linestyle="-")

plothus(NACAsymplot,
        airfoil_df["NACA 0018 X"],
        airfoil_df["NACA 0018 Chord"],
        datalabel='NACA 0018 Chord Line',
        linestyle="--")

plt.axis('equal')


x, upper, lower, chord = NACAThicknessEquation(2, 4, 18, 100)

airfoil_df["NACA 2418 X"], airfoil_df["NACA 2418 Upper"] = x, upper
airfoil_df["NACA 2418 Lower"], airfoil_df["NACA 2418 Chord"] = lower, chord


fig, NACAplot = plt.subplots()
plothusly(NACAplot,
          airfoil_df["NACA 2418 X"],
          airfoil_df["NACA 2418 Upper"],
          xtitle=r'Non-dimensionalized x-position $\frac{x}{c}$',
          ytitle=r'Non-dimensionalized z-position $\frac{z}{c}$',
          datalabel="NACA 2418 Upper Surface",
          title="NACA 2418 Airfoil Plot")

plothus(NACAplot,
        airfoil_df["NACA 2418 X"],
        airfoil_df["NACA 2418 Lower"],
        datalabel='NACA 2418 Lower Surface',
        linestyle="-")

plothus(NACAplot,
        airfoil_df["NACA 2418 X"],
        airfoil_df["NACA 2418 Chord"],
        datalabel='NACA 2418 Chord Line',
        linestyle="--")

plt.axis('equal')

#%%###########################

# Problem 2



wakeveldat = np.loadtxt('Data/WakeVelDist.dat', delimiter=',')
column_names = ["y/c", "u1", "u2"]
wake_velocity_dat = pd.DataFrame(wakeveldat, columns=column_names)
rho = 1.2

sectional_drag = MomentumDeficit(wake_velocity_dat["y/c"],
                                 wake_velocity_dat["u1"],
                                 wake_velocity_dat["u2"],
                                 rho)

coeffs = np.polyfit(wake_velocity_dat["y/c"], wake_velocity_dat["u2"], 6)

yc = np.linspace(-1.25, 1.25, 100)
u2 = 0*yc

coeffs = np.flip(coeffs)

for k in range(len(coeffs)):
    u2 += coeffs[k] * (yc**k)

sectional_drag_2 = MomentumDeficit(yc, 1, u2, rho)

fig, test = plt.subplots()
plothusly(test,
          wake_velocity_dat["u2"],
          wake_velocity_dat["y/c"],
          datalabel='Original',
          xtitle=r'Fluid Exit velocity u$_2$',
          ytitle=r'Non-dimensionalized y dimension $\frac{y}{c}$')
plothus(test, u2, yc, datalabel=r'6$^{th}$ Degree Polynomial Fit')
plt.xlim([0, 1])
plt.ylim([-1.5, 1.5])

#%%###########################

# Problem 3

xfoildat = pd.read_csv("Data/JRCairfoildataProject1.csv")
print(xfoildat)

# NACA 2418, alpha = 0

# Viscid
airfoilflatHIlo0 = np.loadtxt("Data/naca2418/naca2418ReHI0lower.text", skiprows=1)
airfoilflatHIhi0 = np.loadtxt("Data/naca2418/naca2418ReHI0upper.text", skiprows=1)
airfoilflatHIlo11 = np.loadtxt("Data/naca2418/naca2418ReHI11lower.text", skiprows=1)
airfoilflatHIhi11 = np.loadtxt("Data/naca2418/naca2418ReHI11upper.text", skiprows=1)

airfoilcurveHIlo0 = np.loadtxt("Data/naca0018/naca0018ReHI0lower.text", skiprows=1)
airfoilcurveHIhi0 = np.loadtxt("Data/naca0018/naca0018ReHI0upper.text", skiprows=1)
airfoilcurveHIlo11 = np.loadtxt("Data/naca0018/naca0018ReHI11lower.text", skiprows=1)
airfoilcurveHIhi11 = np.loadtxt("Data/naca0018/naca0018ReHI11upper.text", skiprows=1)



alpha = 0
fig, presplot0 = plt.subplots()

naca = '0018'
plothusly(presplot0,
          airfoilflatHIlo0[:, 0],
          airfoilflatHIlo0[:, 1],
          datalabel=fr'NACA{naca}, lower surface',
          xtitle=r'Non-dimensionalized x-position $\frac{x}{c}$',
          ytitle=r'C$_P$',
          title=fr'Airfoil Pressure Distrubution Comparison at $\alpha$ = {alpha}')

plothus(presplot0,
        airfoilflatHIhi0[:, 0],
        airfoilflatHIhi0[:, 1],
        datalabel=fr'NACA{naca}, upper surface'
        )

naca='2418'
plothus(presplot0,
        airfoilcurveHIlo0[:, 0],
        airfoilcurveHIlo0[:, 1],
        datalabel=fr'NACA{naca}, lower surface',
        )

plothus(presplot0,
        airfoilcurveHIhi0[:, 0],
        airfoilcurveHIhi0[:, 1],
        datalabel=fr'NACA{naca}, upper surface'
        )
plt.gca().invert_yaxis()

alpha = 11
fig, presplot11 = plt.subplots()

naca = '0018'
plothusly(presplot11,
          airfoilflatHIlo11[:, 0],
          airfoilflatHIlo11[:, 1],
          datalabel=fr'NACA{naca}, lower surface',
          xtitle=r'Non-dimensionalized x-position $\frac{x}{c}$',
          ytitle=r'C$_P$',
          title=fr'Airfoil Pressure Distrubution Comparison at $\alpha$ = {alpha}')

plothus(presplot11,
        airfoilflatHIhi11[:, 0],
        airfoilflatHIhi11[:, 1],
        datalabel=fr'NACA{naca}, upper surface'
        )

naca='2418'
plothus(presplot11,
        airfoilcurveHIlo11[:, 0],
        airfoilcurveHIlo11[:, 1],
        datalabel=fr'NACA{naca}, lower surface',
        )

plothus(presplot11,
        airfoilcurveHIhi11[:, 0],
        airfoilcurveHIhi11[:, 1],
        datalabel=fr'NACA{naca}, upper surface'
        )
plt.gca().invert_yaxis()



#%%###########################

naca='0018'; alpha = 11

# Inviscid

airfoilflatLOlo = np.loadtxt("Data/naca0018/naca0018ReLO11lower.text", skiprows=1)
airfoilflatLOhi = np.loadtxt("Data/naca0018/naca0018ReLO11upper.text", skiprows=1)

# Viscid
airfoilflatHIlo = np.loadtxt("Data/naca0018/naca0018ReHI11lower.text", skiprows=1)
airfoilflatHIhi = np.loadtxt("Data/naca0018/naca0018ReHI11upper.text", skiprows=1)



fig, presplot4 = plt.subplots()

plothusly(presplot4,
          airfoilflatLOlo[:, 0],
          airfoilflatLOlo[:, 1],
          datalabel='Lower, Inviscid',
          xtitle=r'Non-dimensionalized x-position $\frac{x}{c}$',
          ytitle=r'-C$_P$',
          title=fr'NACA {naca} Pressure Distribution at $\alpha$ = {alpha}')

plothus(presplot4,
        airfoilflatLOhi[:, 0],
        airfoilflatLOhi[:, 1],
        datalabel='Upper, Inviscid',
        )

plothus(presplot4,
        airfoilflatHIlo[:, 0],
        airfoilflatHIlo[:, 1],
        datalabel='Lower, Re=6e5',
        )

plothus(presplot4,
        airfoilflatHIhi[:, 0],
        airfoilflatHIhi[:, 1],
        datalabel='Upper, Re=6e5',
        )
plt.gca().invert_yaxis()

#%%###########################

# Problem 3.3

airfoilcurveLOlo11 = np.loadtxt("Data/naca2418/naca2418ReLO11lower.text", skiprows=1)
airfoilcurveLOhi11 = np.loadtxt("Data/naca2418/naca2418ReLO11upper.text", skiprows=1)


# x, upper, lower, chord = NACAThicknessEquation(2, 4, 18, 0, use_other_x_points=)


#%%###########################

# Problem 4

# Problem 1.15

n = 250

vel_list = np.linspace(70, 250, n)

cessna_skylane = Airplane115(174, 2950, 0.002377)

c_l_vec = np.zeros((n, 1))
c_d_vec = np.zeros((n, 1))
ld_vec = np.zeros((n, 1))

for i in range(len(vel_list)):
    c_l_vec[i], c_d_vec[i], ld_vec[i] = cessna_skylane.getProperties(vel_list[i])
    
fig, c_l_plot = plt.subplots()
plothusly(c_l_plot,
          vel_list,
          c_l_vec,
          xtitle="Flight Velocity (ft/s)",
          ytitle=r'Lift Coefficient C$_L$',
          title=r'Unidentified Aircraft C$_L$ with varying speed',
          datalabel='Unidentified Aircraft')

    
fig, c_d_plot = plt.subplots()
plothusly(c_d_plot,
          vel_list,
          c_d_vec,
          xtitle="Flight Velocity (ft/s)",
          ytitle=r'Drag Coefficient C$_D$',
          title=r'Unidentified Aircraft C$_D$ with varying speed',
          datalabel='Unidentified Aircraft')

fig, ldplot = plt.subplots()
plothusly(ldplot,
          vel_list,
          ld_vec,
          xtitle="Flight Velocity (ft/s)",
          ytitle=r'Lift/Drag Ratio L/D',
          title=r'Unidentified Aircraft L/D with varying speed',
          datalabel='Unidentified Aircraft')

#%%###########################

# Problem 3.11

LAMBDA = 10  # Capital lambda, source strength

# Note: np.log is natural log
source_flow_phi = lambda r: LAMBDA/(2*np.pi) * np.log(r)
source_flow_psi = lambda theta: LAMBDA/(2*np.pi) * theta

data_points = np.linspace(1, 10, 1000000)


laplace_verified_phi = np.gradient(data_points * np.gradient(source_flow_phi(data_points))) / data_points
laplace_verified_psi = np.gradient(np.gradient(source_flow_psi(data_points))) / (data_points**2)

max_phi = max(laplace_verified_phi)
max_psi = max(laplace_verified_psi)

print(f'Max phi = {max_phi}')
print(f'Max psi = {max_psi}')
#%%###########################

# 3.16
R = 1
r_list = np.linspace(R, 5, n)
theta_list = np.linspace(0, 2*np.pi, n)

non_lifting_psi = lambda r, theta, V: (V * r * np.sin(theta) * (1- R/(r**2)))

data_storage_lo = np.zeros((n, 3))
data_storage_hi = np.zeros((n, 3))

i = 0
for r, theta in zip(r_list, theta_list):
    data_storage_lo[i, 0] = r
    data_storage_lo[i, 1] = theta
    data_storage_lo[i, 2] = non_lifting_psi(r, theta, 20)
    i += 1

i = 0
for r, theta in zip(r_list, theta_list):
    data_storage_hi[i, 0] = r
    data_storage_hi[i, 1] = theta
    data_storage_hi[i, 2] = non_lifting_psi(r, theta, 40)
    i += 1

