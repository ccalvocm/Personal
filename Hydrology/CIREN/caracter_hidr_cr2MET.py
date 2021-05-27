#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:13:57 2021

@author: faarrosp
"""

import os
import rioxarray as rioxr
import xarray as xr
import rasterio as rio
import rasterio.mask
import geopandas as gpd
from unidecode import unidecode
from matplotlib import pyplot
from rasterstats import zonal_stats, point_query
import fiona
import pandas as pd
# def reproject_nc(src):
    
#     xds = xr.open_dataset(src)
#     xds_reproj = xds.rio.reproject('EPSG:32719', resampling='bilinear')
    
#     return xds_reproj

def raster_mask_by_geom(srcfile, dstfile, geometry):
    
    shapes = [geometry]
        
    with rio.open(srcfile) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True,
                            nodata=-999)
        # possible drivers: JP2OpenJPEG, GTiff
        out_meta = src.meta
        out_meta.update({'driver': 'GTiff',
                        'height': out_image.shape[1],
                        'width': out_image.shape[2],
                        #'dtype': 'float32',
                        'transform': out_transform,
                        'nodata': -999})
        #out_image = out_image.astype(np.float32)
        #out_image[out_image==0]=0
        
        with rio.open(dstfile, 'w', **out_meta) as dest:
            dest.write(out_image)
# --------------------
# Paths
# --------------------

fdpsrc = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'cr2MET')
fpsrc = os.path.join(fdpsrc, 'ysum_EPSG32719.nc')

fdpcuencas = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH')
fpcuencas = os.path.join(fdpcuencas, 'Cuencas',
                         'Cuencas_DARH_2015_AOHIA_ZC.geojson')

fpsubcuencas = os.path.join(fdpcuencas, 'Subcuencas',
                            'SubCuencas_DARH_2015_AOHIA_ZC.geojson')

fdpcamels = os.path.join('..', 'SIG', 'Cuencas_CAMELS')
fpcamels =  os.path.join(fdpcamels,
                        'Cuencas_cabecera_MaipoRapelMataquitoMaule_EPSG32719.shp')


gdfcuencas = gpd.read_file(fpcuencas)
gdfsubcuencas = gpd.read_file(fpsubcuencas)
gdfcabecera = gpd.read_file(fpcamels)
gdfcabecera = gdfcabecera.to_crs('EPSG:32719')

root_cr2 = os.path.join('.', 'outputs', 'caracter_hidr', 'cr2MET')

years = [str(i) for i in range(1979,2021)]
bands = [str(i) for i in range(1,43)]


# -------- Macrocuencas (individual)
for idx in gdfcuencas.index:
    geometry = gdfcuencas.loc[idx,'geometry']
    name = unidecode(gdfcuencas.loc[idx, 'NOM_CUENCA'])
    name = name.replace(' ', '_')
    cod = gdfcuencas.loc[idx, 'COD_CUENCA']
    path = os.path.join(root_cr2, cod + '_' + name)
    # os.system('mkdir ' + path)
    for band, year in zip(bands,years):
        dstfile = os.path.join(path, year + '.tif')
        # raster_mask_by_geom('netcdf:'+fpsrc+':Band' + band, dstfile, geometry)
        # gs = gpd.GeoSeries(geometry, crs = 'EPSG:32719')
        # gd = gpd.GeoDataFrame(gs)
        # gd.to_file(os.path.join(path, 'tmp.shp'))
        
        

#%% Generar dataframe de zonalstats para cuencas subcuencas y cuencas cab
lstcuencas = gdfcuencas['NOM_CUENCA'].values
lstcodigos = gdfcuencas['COD_CUENCA'].values

lstscuencas = gdfsubcuencas['NOM_DGA'].values
lstscodigos = gdfsubcuencas['COD_DGA'].values

lstcabecera = gdfcabecera['gauge_name'].values
lstcodcabec = gdfcabecera['gauge_id'].values

input_rows1 = []
input_rows2 = []
input_rows3 = []

for band, year in zip(bands,years):
    src = fiona.open(fpcuencas)
    zs = zonal_stats(src, 'netcdf:'+fpsrc+':Band' + band, stats=['min', 'max', 'mean'])
    
    for x,cc,cd in zip(zs, lstcuencas,lstcodigos):
        newrow = dict()
        newrow.update(x)
        newrow.update({'NOM_CUENCA': cc,
                       'COD_CUENCA': cd,
                       'Ano': year})
        input_rows1.append(newrow)

    src = fiona.open(fpsubcuencas)
    zs = zonal_stats(src, 'netcdf:'+fpsrc+':Band' + band, stats=['min', 'max', 'mean'])
    
    for x,cc,cd in zip(zs, lstscuencas,lstscodigos):
        newrow = dict()
        newrow.update(x)
        newrow.update({'NOM_CUENCA': cc,
                       'COD_CUENCA': cd,
                       'Ano': year})
        input_rows2.append(newrow)
        
    src = fiona.open(fpcamels)
    zs = zonal_stats(src, 'netcdf:'+fpsrc+':Band' + band, stats=['min', 'max', 'mean'])
    
    for x,cc,cd in zip(zs, lstcabecera,lstcodcabec):
        newrow = dict()
        newrow.update(x)
        newrow.update({'NOM_CUENCA': cc,
                       'COD_CUENCA': cd,
                       'Ano': year})
        input_rows3.append(newrow)
    


dfcuencas = pd.DataFrame(input_rows1)
dfscuencas = pd.DataFrame(input_rows2)
dfcabecera = pd.DataFrame(input_rows3)

#%%
dst = os.path.join(root_cr2,'CR2MET_estadisticas_x_cuenca')
with pd.ExcelWriter(dst + '.xlsx') as writer:  

    dfcuencas.to_excel(writer, sheet_name='Macrocuencas')
    dfscuencas.to_excel(writer, sheet_name='SubcuencasDARH')
    dfcabecera.to_excel(writer, sheet_name = 'Cabecera')