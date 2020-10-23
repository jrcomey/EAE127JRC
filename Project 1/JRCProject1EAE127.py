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
#%###########################

# Objects

#%###########################

# Functions

def NACAThicknessEquation(N, A, CA, num_points):
    """
        Generates a non-dimensionalized NACA airfoil given NACA numbers.

    Parameters
    ----------
    N : Ratio of max camber to chord length
    A : Location of max camber
    CA : Thickness ratio
    num_points : TYPE
        DESCRIPTION.


    Returns
    -------
    x_non_dim_full : List of non-dimenionalized points from 0 to 1 to 0
    z : Airfoil non-dimensionalized z poisition from xc = 0 to 1 to 0
    x_non_dim : Non-dimensionalized x points for the chord line
    zcc : Chord line

    """
    p = 0.1 * A
    m = 0.01 * N
    t = 0.01 * CA
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
    
    # Put both upper and lower edges into one continuous curve

    x_non_dim_full = np.concatenate((x_non_dim, np.flip(x_non_dim)))
    z = np.concatenate((zup, np.flip(zdown)))
    return x_non_dim_full, z, x_non_dim, zcc


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

# Problem 1


x, z, xchord, zchord = NACAThicknessEquation(0, 0, 18, 100)
column_names = ["NACA 0018 X", "NACA 0018 Z"]
airfoil_df = pd.DataFrame(columns=column_names)

airfoil_df["NACA 0018 X"], airfoil_df["NACA 0018 Z"] = x, z



fig, NACAsymplot = plt.subplots()
plothusly(NACAsymplot,
          airfoil_df["NACA 0018 X"],
          airfoil_df["NACA 0018 Z"],
          xtitle='Non-dimensionalized x/c',
          ytitle="Non-dimensionalized z/c",
          datalabel="NACA 0018 Airfoil",
          title="NACA 0018 Airfoil Plot")

plothus(NACAsymplot,
        xchord,
        zchord,
        datalabel='NACA 0018 Chord Line',
        linestyle="--")

plt.axis('equal')


x, z, xchord, zchord = NACAThicknessEquation(2, 4, 18, 100)

airfoil_df["NACA 2418 X"], airfoil_df["NACA 2418 Z"] = x, z


fig, NACAplot = plt.subplots()
plothusly(NACAplot,
          airfoil_df["NACA 2418 X"],
          airfoil_df["NACA 2418 Z"],
          xtitle='Non-dimensionalized x/c',
          ytitle="Non-dimensionalized z/c",
          datalabel="NACA 2418 Airfoil",
          title="NACA 2418 Airfoil Plot")

plothus(NACAplot,
        xchord,
        zchord,
        datalabel='NACA 2418 Chord Line',
        linestyle="--")

plt.axis('equal')


#%%###########################

