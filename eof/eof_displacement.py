#!/usr/bin/env python

import numpy as np
import gamma_to_py as gampy
import read_grid as rg
from PIL import Image, ImageDraw
import os
import stats
import gaps_gen as gg
import copy
import eof_reconstruction as eof_r
import matplotlib.pyplot as plt
import noisegen as ng

# Open displacement maps
path = '/home/hipperta/Documents/img/serie_temp/argentiere/s1a/'

im = []
n_i = 0
for filename in sorted(os.listdir(path)):
    fichier = gampy.ouvrir(path+filename, 's')
    im.append(gampy.reshape(fichier))
    n_i += 1

# Create mask with shape defined in text file (Gamma)
polygon = rg.read_grid('argentiere_mask.txt', 0, 1, False, False)
width = im[0].shape[1]
height = im[0].shape[0]
img = Image.new('L', (width, height), 0)
ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
mask = np.array(img)

im_i = copy.deepcopy(im)

n_pts = len(im[0][mask==True])
datai = np.zeros([n_i, n_pts])
dataii = np.zeros([n_i, n_pts])

# Generate cross validation points
mask_gaps = np.zeros([n_i, n_pts], dtype=bool)
mask_cv = np.zeros([n_i, n_pts], dtype=bool)
mask_g_temp = np.zeros([n_i, n_pts], dtype=bool)

ng = [50]
n_cv = [np.arange(i) for i in ng]

col = False
# Pretreatement of data
for i in range(n_i):
    datai[i] = im[i][mask==True]
datai_mean = stats.compute_mean(datai, n_i, col)
dataii = copy.copy(datai)
dataii = stats.remove_mean(dataii, datai_mean, n_i, col)

displ = copy.copy(datai)

for i in range(n_i):
    mask_gaps[i][im[i][mask==True]==0] = True
    mask_g_temp[i] = copy.copy(mask_gaps[i])
    mask_gaps[i] = gg.gen_cv_mask(mask_gaps[i], n_pts, n_cv[0])

mask_cv = np.logical_xor(mask_gaps, mask_g_temp)

# Apply mask on displacement field
displ = gg.mask_field(displ, mask_gaps, np.nan)
displ_mean = stats.compute_mean(displ, n_i, col)
displ = stats.remove_mean(displ, displ_mean, n_i, col)

displtp = copy.copy(displ)

# Initialization
NOISE_TYPE = 'rand'
mu, sigma = 0, 1
blanc = np.random.normal(mu, sigma, len(displtp[mask_gaps == True]))
if NOISE_TYPE == 'corr':
    expo = 1.4 # exponent in correlation function (as: 1/(r)**expo)
    geo_noise = ng.geo(r, expo)
    noise = ng.gen_noise_series(geo_noise, blanc, nt)
elif NOISE_TYPE == 'rand':
    noise = blanc
mul = [0.1]
noises = [noise*i for i in mul]

displtp[mask_gaps == True] = 0#noises[0]

# Reconstruction procedure
displs = []
displtp = displtp.T
dataii = dataii.T
mask_gaps = mask_gaps.T
mask_cv = mask_cv.T
displs.append(eof_r.reconstruct_field(displtp, dataii, mask_gaps, mask_cv, i))

for i in range(len(displs)):
    displs[i] = displs[i].T
    displs[i] = stats.add_mean(displs[i], displ_mean, n_i, False)
    
for i in range(n_i):
    im[i][mask==1] = displs[len(displs)-1][i]
    im[i][mask==0] = np.nan


indices = range(5)
#ind = 10
for ind in indices : 
    im_i[ind][mask==0] = np.nan

    img_sl = im_i[ind][1000:1450,900:1200]
    img_sl[img_sl==0.] = np.nan

    img_slice = im[ind][1000:1450,900:1200]
    cmaps = ['RdGy', 'inferno', 'viridis', 'seismic']

    mat = [img_sl, img_slice]
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 5))
    min1, max1 = np.nanmin(img_sl), np.nanmax(img_slice)
    j=0
    plt.suptitle('12 days-displacement [m] in radar LOS', fontsize=18) 
    for ax in axs.flat:
        image = ax.imshow(mat[j], vmin=min1, vmax=max1, cmap = cmaps[2])
        j+=1
        fig.subplots_adjust(right=0.8)
        cb_ax = fig.add_axes([0.84, 0.11, 0.022, 0.76])
        cbar = fig.colorbar(image, cax=cb_ax)


plt.show()




