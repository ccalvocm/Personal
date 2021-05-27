#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:04:09 2021

@author: faarrosp
"""

import os
import rioxarray
import xarray
from pyproj import CRS
import rasterio

west = -72.7000030649119395
south = -36.6000007733906827
east = -69.7500015127247792
north = -32.8999992266093244
width, height = 59, 74
aff = rasterio.transform.from_bounds(west, south,
                                     east, north,
                                     width, height)



os.chdir('/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
         'AOHIA_ZC/Etapa 1 y 2/GIS/cr2MET')
    
xds = xarray.open_dataset("CR2MET_crop_copy.nc")

cc = CRS.from_epsg(4326)

xds.rio.write_crs(cc.to_string(), grid_mapping_name = 'latitude_longitude', inplace=True)
xds.rio.write_grid_mapping(grid_mapping_name='latitude_longitude', inplace=True)
xds.rio.write_transform(transform=aff, grid_mapping_name='latitude_longitude', inplace=True)

xds.pr.rio.to_raster("cr2MET_coords.nc", driver = 'NetCDF')


#%%
xds.close()