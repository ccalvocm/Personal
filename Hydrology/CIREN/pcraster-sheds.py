#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:47:24 2020

@author: felipe
"""

import os
import pcraster as pcr
from matplotlib import pyplot as plt
import numpy as np
from pysheds.grid import Grid
import seaborn as sns
import warnings
import matplotlib.colors as colors
import rasterio

# importar un dem transformado a map
os.chdir('../Etapa 1 y 2/DEM/masked_DEMs/')

cuencas = ['Rio_Cortaderal']
files = []

folderfiles = [x for x in os.listdir() if (x.endswith('.map') and 'ldd' not in x)]

for name in cuencas:
    for file in folderfiles:
        if name in file:
            files.append(file)
        else:
            pass
    
for file in files:

    # leer el dem en pcraster
    dem_map = pcr.readmap(file)
    pcr.setclone(file)
    
    # crear el LDD
    ldd = pcr.lddcreate(dem_map, 9999999,9999999,9999999,9999999)
    pcr.aguila(ldd)
    
    # revisar red de cauces
    strahler = pcr.streamorder(ldd)
    pcr.aguila(strahler)
    
    # guardar el ldd como ldd.map
    pcr.report(ldd, file[:-4] + '_ldd' + '.map')
    # transformar el ldd.map en ldd.asc para que lo lea pySheds
    os.system(' '.join(['map2asc -a -m 9999',
                        file[:-4] + '_ldd' + '.map',
                        file[:-4] + '_ldd' + '.asc']))
    
    # crear objeto Grid
    grid = Grid()
    # importar el ascii (.asc)
    grid.read_ascii(file[:-4] + '_ldd' + '.asc', data_name='dir')
    # plotear
    plt.imshow(grid.view('dir'))
    # dirmap de Pcraster
    #N    NE    E    SE    S    SW    W    NW
    dirmap = (8,  9,  6,   3,    2,   1,    4,  7)
    
    # revisar acumulacion de flujo
    grid.accumulation(data='dir', dirmap=dirmap, out_name='acc',
                      routing = 'd8')
    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_alpha(0)
    plt.grid('on', zorder=0)
    acc_img = np.where(grid.mask, grid.acc + 1, np.nan)
    im = ax.imshow(acc_img, extent=grid.extent, zorder=2,
                   cmap='cubehelix',
                   norm=colors.LogNorm(1, grid.acc.max()))
    plt.colorbar(im, ax=ax, label='Upstream Cells')
    plt.title('Flow Accumulation')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # leer el raster
    rasterpath = file[:-4] + '.tif'
    
    dataset = rasterio.open(file)
    plt.imshow(dataset.read(1), cmap='pink')
    
    
