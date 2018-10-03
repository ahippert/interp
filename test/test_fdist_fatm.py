#
# Tests of fdist and fatm modules
# 
# AH 05/2018

import numpy as np
import matplotlib.pyplot as plt
import pylab
import scipy as scp
from numpy import sin,cos,pi,exp
from time import clock
from matplotlib import rc

from fdist import *
from fatm import *

nx, ny = 500, 500 # Résolution spatiale (x,y) = (colonnes, lignes)
nt = 20 # Résolution temporelles (nombre d'images)
"""Quelques donnée préliminaires"""
x, y = np.meshgrid(np.linspace(-1,1,nx), np.linspace(-1,1,ny)) #Grillage de l'espace
r = np.sqrt(x**2+y**2) # Grillage des distances à l'origine

# Cas réel
Fdist = genere_topo(nt, r, lambda r, t:volcan(r,t+30/nt)) #Premier type
#Fdist = genere_topo(volcan2) #Second type
#Fdist = genere_topo(volcan3) #Troisième type

"""Bruit atmosphérique"""
blanc = np.random.standard_normal((nt,nx,ny)) # Bruit blanc de base
Fatm = genere_atm(geo(r), blanc, nt)*3

Ftot = Fdist+Fatm

mat = [Fdist, Fatm, Ftot]
for i in mat :
    plt.figure()
    plt.imshow(i[0])
plt.show()
