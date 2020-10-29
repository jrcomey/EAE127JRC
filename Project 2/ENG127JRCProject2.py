#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:57:10 2020

@author: jack

EAE 127 Project 2

Stream sources and 
"""

# Imports 

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("classic")
#%###########################

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
    
    def GetStrength(self):
        return self.strength
        
#%###########################

# Functions

#%###########################

# Main

source_x_list = np.array([0])
source_y_list = np.array([0, 0, 1, -1])
source_l_list = np.array([2.1])

u_inf = 0.6
v_inf = 0

N = 1000

x_bounds = np.linspace(-4, 4, N)
y_bounds = np.linspace(-3, 3, N)

X, Y = np.meshgrid(x_bounds, y_bounds)


n = len(source_x_list)
source_sink_objs = np.empty((n, 1), dtype=object)

i = 0
for x, y, l in zip(source_x_list, source_y_list, source_l_list):
    source_sink_objs[i] = Source(x, y, l)
    i += 1
    
u_freestream = u_inf * np.ones((N, N))
v_freestream = v_inf * np.ones((N, N))

u_total = np.zeros((N, N))
v_total = np.zeros((N, N))

u_total += u_freestream
v_total += v_freestream

for i in range(len(source_sink_objs)):
    u_source , v_source = source_sink_objs[i].item().GetVelocity(X, Y)
    u_total += u_source
    v_total += v_source

x_source = np.zeros((len(np.where(source_l_list > 0)[0]), 1))
y_source= np.zeros((len(np.where(source_l_list > 0)[0]), 1))
x_sink = np.zeros((len(np.where(source_l_list < 0)[0]), 1))
y_sink = np.zeros((len(np.where(source_l_list < 0)[0]), 1))

k = 0
for i in zip(*np.where(source_l_list > 0)):
    x_source[k], y_source[k] = source_x_list[i], source_y_list[i]
    k += 1
    

k = 0
for i in zip(*np.where(source_l_list < 0)):
    x_sink[k], y_sink[k] = source_x_list[i], source_y_list[i]
    k += 1



fig, streamplot = plt.subplots()
plt.grid(True)
plt.xlabel("X Direction")
plt.ylabel("Y Direction")
plt.streamplot(X, Y, u_total, v_total, 
               density = 2, linewidth = 0.4, arrowsize=0.5,
               color='blue')
plt.title("Stream Plot")

plt.scatter(x_source, y_source, s=40, marker='o', color='red',label='Sources')
plt.scatter(x_sink, y_sink, s=40, marker='o', color='black', label="Sinks")
plt.legend(loc='upper right')
plt.axis('equal')