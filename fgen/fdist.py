# coding=utf-8
import numpy as np
from numpy import sin,cos,pi,exp
#import var as v


def volcan(r, t, r0 = 0, e = 2):
    z = (1 - np.abs(r0-r)/e)*t 
    return z*(z>0)

def volcan2(r,t, modes):
    f = 1
    f1 = 3
    if modes >=2:
        z = sin(f*pi/2*t)*cos(f*r*pi/2)
    if modes >=3:
        z += cos(f*3*pi/2*t)*0.5*cos(f*5*r*pi)
    if modes >=4:
        z += sin(f*5*pi/2*t)*0.1*cos(f*10*r*pi)
    if modes >=5:
        z += sin(f*7*pi/2*t)*0.3*sin(f*15*r*pi)
    if modes >=6:
        z += sin(f*pi/2*t)*0.1*sin(f*r*pi)
    if modes >=7:
        z += cos(f1*7*pi/2*t)*0.3*cos(f1*11*r*pi)
    return z
    
def volcan3(r, t, r0=0, e=2):
    z=(1-np.abs(r0-r)/e)*t + 0.5*(1-np.abs(r0-r)/e)*t**2
    return z*(z>0)
    
def volcan4(r, t, r0=0, e=2):
    z = np.abs(1 - np.abs(r0-r)/e)*t + sin(pi/2*t)*cos(r*pi/2)
    return z*(z>0)

def depla(r,t):
    return sin(2*pi*(r+t))
    
def genere_topo(nt, r, topo):
    return np.array([ topo(r,(t)/(nt)) for t in range(10,nt+10)])
    
    
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
