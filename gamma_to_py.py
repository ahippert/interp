#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def ouvrir(fichier, nature):
    f = open(fichier, 'rb')
    a = np.fromfile(f, dtype='>f')

    if nature == "cpx":
        reel = a[::2]
        imag = a[1::2]
        a = reel + 1j*imag

    f.close()
    return a

def reshape(a, col = 6468):
    n = a.shape[0]
    ligne = n//col
    res = n%col

    if res != 0:
        a = a[:ligne*col]

    return a.reshape((ligne, col))
