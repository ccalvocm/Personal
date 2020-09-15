# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:47:22 2020

@author: Carlos
"""
import os
from osgeo import gdal
import numpy as np

# Directorios

folder = r'C:\Users\Carlos\Documents\Python Scripts'
os.chdir(folder)
dst_dir = r'C:\Users\Carlos\Documents\Python Scripts'

# Funciones


def hdf_subdataset_extraction(file_name, dst_dir):
    #"""Descomprime el HDF, calcula el NDSI y proyecta de sinusoidales a WGS84"""
    # Leer HDF
    hdf_ds = gdal.Open(file_name, gdal.GA_ReadOnly)
    band_ds = gdal.Open(hdf_ds.GetSubDatasets()[3][0], gdal.GA_ReadOnly)

    # Leer como arreglo Numpy
    band_array = band_ds.ReadAsArray().astype(np.float32)

    # filtrar celdas sin datos
    band_array[band_array == -28672] = -32768
    
    #Filtrar por NDSI de nieve
    band_array[band_array < 0.4] = -32768

    # nombre de archivo
    band_path = os.path.join(dst_dir, os.path.basename(os.path.splitext(file_name)[0]) + "-sd" + str(3+1) + ".tif")

    # Crear Geotiff
    out_ds = gdal.GetDriverByName('GTiff').Create(band_path,band_ds.RasterXSize,band_ds.RasterYSize, 1,gdal.GDT_Int16, ['COMPRESS=LZW', 'TILED=YES'])
    out_ds.SetGeoTransform(band_ds.GetGeoTransform())
    out_ds.SetProjection(band_ds.GetProjection()) 
    out_ds.GetRasterBand(1).WriteArray(band_array)
    out_ds.GetRasterBand(1).SetNoDataValue(-32768)
    out_ds = gdal.Warp(band_path,out_ds,
          dstSRS='EPSG:4326' ,
          format = 'GTiff',
          cutlineLayer = 'extent',
          dstNodata = 0)

    out_ds = None  #Cerrar y guardar
    
# Main

def main():
    
    for file in os.listdir(folder):
        if file.endswith('.hdf'):
            hdf_subdataset_extraction(file,folder)
            

if __name__ == '__main__':
    main()

