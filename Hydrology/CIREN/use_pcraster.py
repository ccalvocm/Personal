#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:36:55 2020

@author: felipe
"""

# use pcraster

import pcraster as pcr
import os
import unidecode

#%%
path = '../Etapa 1 y 2/DEM/masked_DEMs/'
files = [x for x in os.listdir(path) if x.endswith('.tif')]
path = path.replace(" ", "\\ ")

for file in files:
    # file = unidecode.unidecode(file)
    
    file = file.replace(" ", "\\ ")
    os.system(' '.join(['gdal_translate', '-ot Float32',
                        '-of PCRaster', '-mo PCRASTER_VALUESCALE=VS_SCALAR',
                        '-a_nodata 9999',
                        path + file, path + file[:-4] + '.map']))
    
#%% Use PCRaster
path = '../Etapa 1 y 2/DEM/masked_DEMs/'

files = [x for x in os.listdir(path) if x.endswith('.map')]
map_dem = pcr.readmap(path + files[0])
# pcr.aguila(map_dem)
ldd = pcr.lddcreate(map_dem, 9999999,9999999,9999999,9999999)
new_dem = pcr.lddcreatedem(map_dem,9999999,9999999,9999999,9999999)
# pcr.aguila(ldd)
strahler = pcr.streamorder(ldd)
x = 305645
y = 6.23308e6
pcr.report(ldd, path + 'ldd.map')
os.system(' '.join(['map2asc', path + 'new_' + files[0],
                    path + 'new_' + files[0][:-4] + '.asc']))


points = pcr.readmap(path + 'points.map')

catchment = pcr.catchment(ldd,points)

