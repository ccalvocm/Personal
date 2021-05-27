#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:01:01 2020

@author: felipe
"""
# importar librerias
import geopandas
import rasterio
import rasterio.mask
from matplotlib import pyplot as plt
import unidecode
import matplotlib.colors as colors
from pysheds.grid import Grid
import numpy as np


# raster original

# path_dem = '../Etapa 1 y 2/DEM/masked_DEMs/Rapel_maskedF32.tif'
# path_shps = '../Etapa 1 y 2/GIS/Cuencas_DARH/masked_cuencas/Rapel_subcuenca.shp'

path_dem = '../Etapa 1 y 2/DEM/DEM Alos 5a a 8a mar.jp2'
path_shps = '../Etapa 1 y 2/GIS/Cuencas_DARH/Cuencas/Cuencas_DARH_2015.shp'

subcuencas = geopandas.read_file(path_shps)
subcuencas = subcuencas[subcuencas['COD_CUENCA'] == '1300']


with rasterio.open(path_dem) as src:
    for cuenca in subcuencas.index:
        
        # identificador = subcuencas.loc[cuenca, 'COD_DGA'] + '_' + \
        #     unidecode.unidecode(subcuencas.loc[cuenca, 'NOM_DGA'])
        
        geom = [subcuencas.loc[cuenca, 'geometry']]
        out_image, out_transform = rasterio.mask.mask(src, geom, crop=True,
                                                      filled = True, nodata = 9999)
        out_image = out_image.astype(np.float32)
        out_image[out_image == 9999] = np.nan
        
        
        # savepath = '../Etapa 1 y 2/DEM/masked_DEMs/' + identificador
        savepath = '../Etapa 1 y 2/DEM/masked_DEMs/' + 'Rio_Maipo'
        
        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform,
                 "dtype": 'float32',
                 "nodata": np.nan})
        
        
        with rasterio.open(savepath + ".tif", "w", **out_meta) as dest:
            dest.write(out_image)
            
        # with rasterio.open(savepath + ".tif") as src:
        #     grid = Grid.from_raster(savepath + '.tif', data_name='dem')
        #     plt.imshow(grid.view('dem'), cmap = 'cubehelix')
            
        #     depressions = grid.detect_depressions('dem')
        #     plt.imshow(depressions)
            
        #     pits = grid.detect_pits('dem')
        #     plt.imshow(pits)
            
        #     grid.fill_depressions(data='dem', out_name='flooded_dem', inplace = True)
        #     depressions = grid.detect_depressions('flooded_dem')
        #     plt.imshow(depressions)
            
        #     flats = grid.detect_flats('flooded_dem')
        #     plt.imshow(flats)
            
        #     grid.resolve_flats(data='dem', out_name='inflated_dem', inplace = True)

        #     flats = grid.detect_flats('inflated_dem')
        #     plt.imshow(flats)
            
        #     grid.resolve_flats(data='inflated_dem', out_name='inflated_dem2', inplace = True)

        #     flats = grid.detect_flats('inflated_dem2')
        #     plt.imshow(flats)
            
        #     # pcraster default dirmap
        #     #N    NE    E    SE    S    SW    W    NW
        #     # dirmap = (8,  9,  6,   3,    2,   1,    4,  7)
            
        #     # pysehds Default dirmap
        #     #N    NE    E    SE    S    SW    W    NW
        #     dirmap = (64,  128,  1,   2,    4,   8,    16,  32)

        #     grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap)
            
        #     fig = plt.figure(figsize=(8,6))
        #     fig.patch.set_alpha(0)
            
        #     plt.imshow(grid.dir, extent=grid.extent, cmap='viridis', zorder=2)
        #     boundaries = ([0] + sorted(list(dirmap)))
        #     plt.colorbar(boundaries= boundaries,
        #                  values=sorted(dirmap))
        #     plt.xlabel('Longitude')
        #     plt.ylabel('Latitude')
        #     plt.title('Flow direction grid')
        #     plt.grid(zorder=-1)
        #     plt.tight_layout()            
            
        #     grid.accumulation(data='dir', dirmap=dirmap, out_name='acc',
        #           routing = 'd8')
        #     fig, ax = plt.subplots(figsize=(8,6))
        #     fig.patch.set_alpha(0)
        #     plt.grid('on', zorder=0)
        #     acc_img = np.where(grid.mask, grid.acc + 1, np.nan)
        #     im = ax.imshow(acc_img, extent=grid.extent, zorder=2,
        #                    cmap='cubehelix',
        #                    norm=colors.LogNorm(1, grid.acc.max()))
        #     plt.colorbar(im, ax=ax, label='Upstream Cells')
        #     plt.title('Flow Accumulation')
        #     plt.xlabel('Longitude')
        #     plt.ylabel('Latitude')
