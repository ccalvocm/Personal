# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:47:22 2020

@author: Carlos
"""
import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from osgeo import osr

# Directorios

folder = r'E:\UC\Corfo\MODIS'
os.chdir(folder)
dst_dir = r'E:\UC\Corfo\MODIS'

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
    out_ds = gdal.GetDriverByName('GTiff').Create(band_path,band_ds.RasterXSize,band_ds.RasterYSize, 1, gdal.GDT_Int16, ['COMPRESS=LZW', 'TILED=YES'])
    out_ds.SetGeoTransform(band_ds.GetGeoTransform())
    out_ds.SetProjection(band_ds.GetProjection()) 
    out_ds.GetRasterBand(1).WriteArray(band_array)
    out_ds.GetRasterBand(1).SetNoDataValue(-32768)
    out_ds = None  #Cerrar y guardar
   
    shp_clip = 'C:\\Users\\Carlos\\Documents\\Python Scripts\\Mask.shp'

    band_path_2 = os.path.join(dst_dir, os.path.basename(os.path.splitext(file_name)[0]) + "-sd" + str(3+1) + "_WGS84.tif")

    out_ds2 = gdal.Warp(band_path_2,band_path,
          dstSRS='EPSG:4326' ,
          format = 'GTiff',
          dstNodata = -32768,
          cutlineLayer = 'Mask',
          cutlineDSName = shp_clip, 
          cropToCutline=True)
    
    out_ds2 = None  #Cerrar y guardar

    os.remove(band_path)
    
    return band_path_2
# Main

def main():

    # Grilla

    ruta_grilla = r'C:\Users\Carlos\Documents\QGIS\Salar\WEAP_Salar_de_Atacama\layers\Grilla.tif'
    grilla = gdal.Open(ruta_grilla)
    band = grilla.GetRasterBand(1)
    arr = band.ReadAsArray()
    arr[:] = 190
        
    for file in os.listdir(folder):
        if file.endswith('.hdf'):
            
            ruta_nieve = hdf_subdataset_extraction(file,folder)
            cobertura_nieve = gdal.Open(ruta_nieve)
            band_nieve = cobertura_nieve.GetRasterBand(1)
            arreglo_nieve = band_nieve.ReadAsArray()
            arreglo_nieve[arreglo_nieve > -32768] = 1
            arreglo_nieve[arreglo_nieve == -32768] = 0
            arr = arr + arreglo_nieve
            
    arr = arr*255/np.max(arr)
    
    [cols, rows] = arr.shape
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create('E:\\UC\\Corfo\\MODIS\\grilla_contada.tif', rows, cols, 1, gdal.GDT_UInt16)
    outdata.SetGeoTransform(cobertura_nieve.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(cobertura_nieve.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(arr)
    outdata.GetRasterBand(1).SetNoDataValue(-32768)##if you want these values transparent
    outdata.FlushCache() ##saves to disk!!
    outdata = None
    band=None
    ds=None
    
    return arr
            

if __name__ == '__main__':
    main()

