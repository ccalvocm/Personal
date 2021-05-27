#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 23:10:52 2021

@author: faarrosp
"""

import os
from osgeo import gdal
from osgeo import ogr
import Calhypso
import geopandas as gpd 
import rasterio as rio
from matplotlib import pyplot as plt
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
import fiona
import rasterio.mask
from rasterio.plot import show
from rasterio.enums import Resampling
from unidecode import unidecode


def import_subcatchments():
    path_folder= os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH')
    path_macrocuencas =  os.path.join(path_folder, 'Cuencas',
                                      'Cuencas_DARH_2015_AOHIA_ZC.geojson')
    path_subcuencas = os.path.join(path_folder, 'Subcuencas',
                                   'SubCuencas_DARH_2015_AOHIA_ZC.geojson')
    gdfcuencas = gpd.read_file(path_macrocuencas)
    gdfsubcuencas = gpd.read_file(path_subcuencas)
    gdfcuencas['COD_CUENCA'] = gdfcuencas['COD_CUENCA'].astype(int)
    gdfsubcuencas['COD_DGA'] = gdfsubcuencas['COD_DGA'].astype(int)
    
    gdfcuencas['areakm2'] = gdfcuencas.area / 1e6 # area en km2
    gdfsubcuencas['areakm2'] = gdfsubcuencas.area / 1e6 # area en km2
    gdfsubcuencas['NOM_DGA'] = gdfsubcuencas['NOM_DGA'].apply(lambda x: unidecode(x))
    
    return gdfcuencas, gdfsubcuencas

def raster_mask_by_geom_Uint16(srcfile, dstfile, geometry):
    
    shapes = [geometry]
        
    with rio.open(srcfile) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True,
                            nodata=15999)
        # possible drivers: JP2OpenJPEG, GTiff
        out_meta = src.meta
        out_meta.update({'driver': 'GTiff',
                        'height': out_image.shape[1],
                        'width': out_image.shape[2],
                        #'dtype': 'float32',
                        'transform': out_transform,
                        'nodata': 15999})
        #out_image = out_image.astype(np.float32)
        #out_image[out_image==0]=0
        
        with rio.open(dstfile, 'w', **out_meta) as dest:
            dest.write(out_image)

def raster_mask_Uint16(srcfile, dstfile, maskfile):
    with fiona.open(maskfile, 'r') as shapefile:
        shapes = [feature['geometry'] for feature in shapefile]
        
        with rio.open(srcfile) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True,
                                nodata=15999)
            out_meta = src.meta
            out_meta.update({'driver': 'GTiff',
                            'height': out_image.shape[1],
                            'width': out_image.shape[2],
                            'transform': out_transform,
                            'nodata': 15999})
            with rio.open(dstfile, 'w', **out_meta) as dest:
                dest.write(out_image)

# path for DEM
demfolder = os.path.join('..', 'Etapa 1 y 2', 'DEM')
demfp = os.path.join(demfolder, 'DEM Alos 5a a 8a mar.jp2')
dem_masked_dst_folder = os.path.join('./','outputs', 'caracter_hidr', 'dem')
dem_rsed_file = os.path.join(dem_masked_dst_folder, 'DEM_APalsar_AOHIA_ZC_rsed.tif')
dem_masked_file = os.path.join(dem_masked_dst_folder, 'DEM_APalsar_AOHIA_ZC.tif')

gdf_cuencas, gdf_subcuencas = import_subcatchments()

folder = os.path.join('..', 'Etapa 1 y 2', 'datos', 'curvas_hipso')
fp = os.path.join(folder,'temp.shp')
# gdf_cuencas2 = gdf_cuencas.reindex([3,0,1,2], copy=True)
gdf_cuencas.to_file(fp)


# Curva hipsometrica macrocuencas
ext_img = 'jpg'


chyp = Calhypso.get_hypsometric_curves(dem_masked_file, fp , 'COD_CUENCA', 'NOM_CUENCA')

fig, axarray = plt.subplots(figsize = (22,22), nrows = 2, ncols = 2,
                            sharex=False, sharey=False)

# for data, idx, ax in zip(chyp, gdf_cuencas2.index, axarray.flat):
#     area = gdf_cuencas2.loc[idx, 'areakm2']
#     name = gdf_cuencas2.loc[idx, 'NOM_CUENCA']
#     x = data._data[:,0] * area
#     y = data._data[:,1] * (data._max_h - data._min_h) + data._min_h
#     ax.plot(x,y[::-1], label = data._name)
#     ax.set_xlabel('Superficie bajo elevación ($km^2$)')
#     ax.set_ylabel('Elevación (m.s.n.m)')
#     ax.set_title('Cuenca ' + name)
    
#     # names_calhypso.append(data._name)
# plt.suptitle('Curvas Hipsométricas\nMacrocuencas')
# filename = os.path.join(folder, 'Chypso_Macrocuencas.' + ext_img)
# plt.savefig(filename, format=ext_img, bbox_inches = 'tight',
#                     pad_inches = 0.1)
# plt.close()


# Curva hipsometricas por separado - macrocuencas

for data, idx in zip(chyp, gdf_cuencas.index):
    fig, ax = plt.subplots(figsize=(8.5,11))
    area = gdf_cuencas.loc[idx, 'areakm2']
    name = gdf_cuencas.loc[idx, 'NOM_CUENCA']
    x = data._data[:,0] * area
    y = data._data[:,1] * (data._max_h - data._min_h) + data._min_h
    ax.plot(x,y[::-1], label = data._name)
    ax.set_xlabel('Superficie bajo elevación ($km^2$)')
    ax.set_ylabel('Elevación (m.s.n.m)')
    ax.set_title('Curva hipsométrica\nCuenca ' + name)
    
    filename = os.path.join(folder, 'Chypso_macro_' + name + '.' + ext_img)
    plt.savefig(filename, format=ext_img, bbox_inches = 'tight',
                        pad_inches = 0.1)
    # plt.close()




# Curva hipsometrica cuencas de cabecera
# dem_masked_dst_folder = os.path.join('./','outputs', 'caracter_hidr', 'dem')
folder_camels = os.path.join('..', 'SIG', 'Cuencas_CAMELS')
catchments_camels = os.path.join(folder_camels,
                                 'Cuencas_cabecera_MaipoRapelMataquitoMaule_epsg32719.shp')
# dem_camels_masked_file = os.path.join(dem_masked_dst_folder, 'DEM_APalsar_camels.tif')
fp_dem_camels = os.path.join('..', 'Etapa 1 y 2', 'DEM', 'DEM_cabecera.tif')
gdf_camels = gpd.read_file(catchments_camels)

gdf_camels['gauge_name'] = gdf_camels['gauge_name'].apply(lambda x: unidecode(x))
gdf_camels['gauge_id'] = gdf_camels.index.values
gdf_camels['areakm2'] = gdf_camels.area / 1e6 # area en km2


gdf_camels.to_file(fp)
chyp = Calhypso.get_hypsometric_curves(fp_dem_camels, fp , 'gauge_id', 'gauge_name')

# fig, axarray = plt.subplots(figsize = (22,22), nrows = 4, ncols = 3,
#                             sharex=False, sharey=False)

# for data, idx, ax in zip(chyp, gdf_camels.index, axarray.flat[:-1]):
#     area = gdf_camels.loc[idx, 'areakm2']
#     name = gdf_camels.loc[idx, 'gauge_name']
#     x = data._data[:,0] * area
#     y = data._data[:,1] * (data._max_h - data._min_h) + data._min_h
#     ax.plot(x,y[::-1], label = data._name)
#     ax.set_xlabel('Superficie bajo elevación ($km^2$)')
#     ax.set_ylabel('Elevación (m.s.n.m)')
#     ax.set_title(name)
    
#     # names_calhypso.append(data._name)
# plt.suptitle('Curvas Hipsométricas\nCuencas de cabecera')
# filename = os.path.join(folder, 'Chypso_Cabecera.' + ext_img)
# plt.savefig(filename, format=ext_img, bbox_inches = 'tight',
#                     pad_inches = 0.1)
# plt.close()

# Curva hipsometricas por separado - subcuencas cabecera

# for data, idx in zip(chyp, gdf_camels.index):
#     fig, ax = plt.subplots(figsize=(8.5,11))
#     area = gdf_camels.loc[idx, 'areakm2']
#     name = gdf_camels.loc[idx, 'gauge_name']
#     x = data._data[:,0] * area
#     y = data._data[:,1] * (data._max_h - data._min_h) + data._min_h
#     ax.plot(x,y[::-1], label = data._name)
#     ax.set_xlabel('Superficie bajo elevación ($km^2$)')
#     ax.set_ylabel('Elevación (m.s.n.m)')
#     ax.set_title('Curva hipsométrica\nCuenca ' + name)
    
#     filename = os.path.join(folder, 'Chypso_cabec_' + name + '.' + ext_img)
#     plt.savefig(filename, format=ext_img, bbox_inches = 'tight',
#                         pad_inches = 0.1)
#     plt.close()