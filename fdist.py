#!/usr/bin/env python
# coding=utf-8

#######################################
#
# interp.fdist
# 
# Generate synthetics displacement-like signals
# evolving in time and space.
#
# Written by Rémi Prebet, updated and added more
# signals by Alexandre Hippert-Ferrer.
#
# @LISTIC, USMB
#
# Last update : 03/11/2018
#
#######################################


import numpy as np


def volcan(r, t, r0 = 0, e = 2):
    z = (1 - np.abs(r0-r)/e)*t 
    return z*(z>0)


def volcan2(r, t, modes):
    f = 1 # Frequencies
    f1 = 3
    if modes >=2:
        z = np.sin(f*np.pi/2*t)*np.cos(f*r*np.pi/2)
    if modes >=3:
        z += np.cos(f*3*np.pi/2*t)*0.5*np.cos(f*5*r*np.pi)
    if modes >=4:
        z += np.sin(f*5*np.pi/2*t)*0.1*np.cos(f*10*r*np.pi)
    if modes >=5:
        z += np.sin(f*7*np.pi/2*t)*0.3*np.sin(f*15*r*np.pi)
    if modes >=6:
        z += np.sin(f*np.pi/2*t)*0.1*np.sin(f*r*np.pi)
    if modes >=7:
        z += np.cos(f1*7*np.pi/2*t)*0.3*np.cos(f1*11*r*np.pi)
    return z

    
def volcan3(r, t, r0=0, e=2):
    z=(1-np.abs(r0-r)/e)*t + 0.5*(1-np.abs(r0-r)/e)*t**2
    return z*(z>0)

    
def volcan4(r, t, r0=0, e=2):
    z = np.abs(1 - np.abs(r0-r)/e)*t + np.sin(np.pi/2*t)*np.cos(r*np.pi/2)
    return z*(z>0)


def depla(r,t):
    return np.sin(2*np.pi*(r+t))

    
def genere_topo(nt, r, topo):
    return np.array([topo(r,(t)/(nt)) for t in range(10,nt+10)])


### TESTS ###


# # Animation
# import numpy as np
# import matplotlib.pyplot as plt

# xx = np.linspace(-1,1,v.nx)

# Tt = range(0,v.nt)
# for t in Tt:
#     plt.ion()
#     t = t/(v.nt-1)
#     plt.figure(1)
#     plt.clf()
#     plt.plot(xx,volcan3(np.abs(xx),t))
#     plt.axis([-1,1,-2,2])
#     plt.title("t={:2.2f} s".format(t))
#     plt.pause(10**-5)

# ## Affichage à t donné
# import numpy as np
# import matplotlib.pyplot as plt

# xx = np.linspace(-1,1,1000)
# t = 0
# plt.clf()
# plt.plot(xx,volcan3(np.abs(xx),t))
# plt.axis([-1,1,-1.6,1.6])
# plt.tight_layout()
# plt.savefig("sin0.png")

# aff(np.array([depla(v.r,t) for t in np.linspace(0,1,30)]), color=True)
