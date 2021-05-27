#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:04:47 2021

@author: faarrosp
"""
from plot_tools import import_catchments
import os
import subprocess



# import rioxarray # for the extension to load
# import xarray
# import rasterio

# xds = xarray.open_dataset("/home/faarrosp/Downloads/cr2met_t2m_rsed.nc", decode_coords="all")
# xds.rio.write_crs('EPSG:4326', inplace=True)
# newxds = xds.rio.reproject("EPSG:32719", resampling=rasterio.enums.Resampling(1))

# import os

# newxds.t2m.rio.to_raster('/home/faarrosp/Downloads/cr2MET_t2m_reproj.nc')


cuencas, _ = import_catchments()

# transform catchment gdf to epsg 4326 for masking netcdf file
cuencas = cuencas.to_crs('EPSG:4326')

folder_cr2 = os.path.join('..','Etapa 1 y 2', 'GIS', 'cr2MET', 'temperatura')
file_cr2 = 'CR2MET_t2m_v2.0_day_1979_2020_005deg.nc'
fp_cr2 = os.path.join(folder_cr2,file_cr2)

def crop_cr2MET(gdf_mask, fp_cr2, fp_dst = None):
    
    if fp_dst is None:
        fp_dst = fp_cr2[:-3] + '_msk.nc'
    else:
        pass
    fp_cr2 = '"' + fp_cr2 + '"'
    fp_dst = '"' + fp_dst + '"'
    
    offset = 1
    xmin = str(gdf_mask.bounds.minx.min() - offset )
    xmax = str(gdf_mask.bounds.maxx.max() + offset )
    ymin = str(gdf_mask.bounds.miny.min() - offset )
    ymax = str(gdf_mask.bounds.maxy.max() + offset )
    
    coords = ','.join([xmin,xmax,ymin,ymax])
    
    files = ' '.join([fp_cr2, fp_dst])

    command = 'cdo sellonlatbox,' + coords + ' ' + files
    print(command)
    
    # os.system(command)

def reproject_cr2MET(fp_cr2):
    options = ' '.join(['-s_srs EPSG:4326',
                        '-t_srs EPSG:32719',
                        '-r average',
                        '-of GTiff'])
    src = 'NETCDF:' + '"' + fp_cr2 + '"' + ':t2m'
    dst = '"' + fp_cr2[:-3] + '_rsed.tif' + '"'
    print('gdalwarp ' + options + ' ' + src + ' ' + dst)

# timespan 1979-01-01 to 2020-04-30
crop_cr2MET(cuencas, fp_cr2)
# reproject_cr2MET(fp_cr2)



