#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:17:24 2020

@author: felipe
"""
# import libraries and load basic files
import geopandas
import fiona
import rasterio
import rasterio.mask
from shapely.geometry import Polygon
import Calhypso
from matplotlib import pyplot as plt
import unidecode
import numpy as np


def std_dev(xarray, xmean):
    s = np.sqrt((xarray - xmean)**2/(len(xarray)))
    return s[0] 



path_shp = '../Etapa 1 y 2/GIS/Cuencas_DARH/Cuencas/Cuencas_DARH_2015.shp'
path_shp_sub = '../Etapa 1 y 2/GIS/Cuencas_DARH/Subcuencas/SubCuencas_DARH_2015.shp'
path_dem = '../Etapa 1 y 2/DEM/masked_DEMs/Rapel_maskedF32.tif'

gdf_basin = geopandas.read_file(path_shp)
gdf_subbasin = geopandas.read_file(path_shp_sub)

basin_code = '0600'

gdf_basin = gdf_basin[gdf_basin['COD_CUENCA'] == basin_code]
gdf_subbasin = gdf_subbasin[gdf_subbasin['COD_CUENCA'] == basin_code]


#%% Hypsometric curve

dem_masked = '../Etapa 1 y 2/DEM/masked_DEMs/Rapel_maskedF32.tif'
subbasin_masked = '../Etapa 1 y 2/GIS/Cuencas_DARH/masked_cuencas/Rapel_subcuenca.shp'

a = Calhypso.get_hypsometric_curves(dem_masked, subbasin_masked , 'COD_MAP', 'COD_MAP')

subbasin = geopandas.read_file(subbasin_masked)
#%%

fig, axarray = plt.subplots(6,6, figsize = (22,17), constrained_layout=True,
                            sharex = False, sharey = False)
plt.suptitle('Curvas Hipsométricas\nCuenca ' + gdf_basin['NOM_CUENCA'].values[0],
             fontsize = 20)
names_calhypso = []
for data, ax in zip(a, axarray.flat):
    x = data._data[:,0] * gdf_subbasin[gdf_subbasin['COD_MAP'] == int(data._name)].Area_Km2.values
    y = data._data[:,1] * (data._max_h - data._min_h) + data._min_h
    ax.plot(x,y[::-1], label = data._name)
    ax.set_title(gdf_subbasin[gdf_subbasin['COD_MAP'] == int(data._name)]['NOM_DGA'].values[0])
    names_calhypso.append(data._name)
    
# fig.text(0.5, 0, 'Superficie [$km^2$]', ha='center')
# fig.text(0, 0.5, 'Altura [m]', va='center', rotation='vertical')
plt.setp(axarray[-1, :], xlabel='Superficie [$km^2$]')
plt.setp(axarray[:, 0], ylabel='Altura [m]')
plt.show()

#%% escribir datos de salida

for data in a:
    filt = gdf_subbasin['COD_MAP'] == int(data._name)
    codigo_DGA = str(gdf_subbasin[filt]['COD_DGA'].values[0])
    codigo_mapa = str(gdf_subbasin[filt]['COD_MAP'].values[0])
    nombre = unidecode.unidecode(gdf_subbasin[filt]['NOM_DGA'].values[0])
    identificador = '_'.join([codigo_DGA, codigo_mapa, nombre]) + '.txt'
    A = gdf_subbasin[filt].Area_Km2.values[0]
    integral = np.trapz(y,x)
    x = data._data[:,0] * A 
    y = data._data[:,1] * (data._max_h - data._min_h) + data._min_h
    h_bar = data._min_h + np.trapz(y,x)/A
    integral_vector = []
    
    for index, yval in enumerate(y):
        integral_vector.append(np.trapz(y[:index+1], x[:index+1])/integral)
    
    with open('../Etapa 1 y 2/datos/curvas_hipso/' + identificador, 'w') as f:
        f.write('Min. Elevation: ' + str(data._min_h) + '\n')
        f.write('Max. Elevation: ' + str(data._max_h) + '\n')
        f.write('Mean. Elevation: ' + str(h_bar) + '\n')
        f.write('Std. Elevation: ' + str(std_dev(y, y.mean())) + '\n')
        f.write('Elevation [m] % Area below elevation' + '\n')
        for xx, yy, ii in zip(x,y[::-1], integral_vector):
            f.write(' '.join(['\n',
                              str(yy),
                              str(xx/A)]))


