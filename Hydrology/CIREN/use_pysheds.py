#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:47:12 2020

@author: felipe
"""

# use of pySHEDS

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pysheds.grid import Grid
import seaborn as sns
import warnings
import os
from pyproj import Proj

warnings.filterwarnings('ignore')

path = '../Etapa 1 y 2/DEM/masked_DEMs/'

files = [x for x in os.listdir(path) if x.endswith('.tif')]

# grid = Grid.from_raster('../Etapa 1 y 2/DEM/masked_DEMs/ldd.tif', data_name = 'dem', nodata = 1e31)
grid = Grid.from_ascii('../Etapa 1 y 2/DEM/masked_DEMs/ldd.asc', data_name='dir')

fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

plt.imshow(grid.view('dir', nodata=9999), extent=grid.extent, cmap='cubehelix', zorder=1)
plt.colorbar(label='Elevation (m)')
plt.grid(zorder=0)
plt.title('Digital elevation map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()

#%% Detect pits
pits = grid.detect_pits('dem')
plt.imshow(pits)

#%% Detect Depressions
depressions = grid.detect_depressions('dem')
plt.imshow(depressions)

#%% Fill Depressions

grid.fill_depressions(data='dem', out_name='flooded_dem', inplace = True)
depressions = grid.detect_depressions('flooded_dem')
plt.imshow(depressions)

#%% Detect Flats
flats = grid.detect_flats('dem')
plt.imshow(flats)
#%%
grid.resolve_flats(data='dem', out_name='inflated_dem', inplace = True)

flats = grid.detect_flats('inflated_dem')
plt.imshow(flats)

#%%
plt.imshow(grid.view('dem'))
#%% coming from PCraster
grid = Grid()
grid.read_ascii('../Etapa 1 y 2/DEM/masked_DEMs/new_prueba.asc', data_name='dem')



#%%
#,
#                      nodata_out=9999)
# grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')#,
                  #nodata_in = 9999)
# pysehds Default dirmap
#N    NE    E    SE    S    SW    W    NW
#dirmap = (64,  128,  1,   2,    4,   8,    16,  32)

# pcraster default dirmap
#N    NE    E    SE    S    SW    W    NW
dirmap = (8,  9,  6,   3,    2,   1,    4,  7)

grid.flowdir(data='dem', out_name='dir', dirmap=dirmap,
             nodata_in = 9999)#,
             # as_crs = Proj('epsg:32719'))

fig = plt.figure(figsize=(8,6))
fig.patch.set_alpha(0)

plt.imshow(grid.dir, extent=grid.extent, cmap='viridis', zorder=2)
boundaries = ([0] + sorted(list(dirmap)))
plt.colorbar(boundaries= boundaries,
             values=sorted(dirmap))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Flow direction grid')
plt.grid(zorder=-1)
plt.tight_layout()


#%% See accumulation
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

#%%
branches = grid.extract_river_network(fdir='dir', acc='acc',
                                      threshold=50, dirmap=dirmap)

for branch in branches['features']:
    line = np.asarray(branch['geometry']['coordinates'])
    plt.plot(line[:, 0], line[:, 1])

#%%
# delineate catchment

# Specify pour point
x, y = 305645, 6233080

# Delineate the catchment
grid.catchment(data='dir', x=x, y=y, dirmap=dirmap, out_name='catch',
               recursionlimit=15000, xytype='label', nodata_out=0, 
               nodata_in = 0)

catch = grid.view('catch', nodata=np.nan)

fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

plt.grid('on', zorder=0)
im = ax.imshow(catch, extent=grid.extent, zorder=1, cmap='viridis')
plt.colorbar(im, ax=ax, boundaries=boundaries, values=sorted(dirmap), label='Flow Direction')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Delineated Catchment')

