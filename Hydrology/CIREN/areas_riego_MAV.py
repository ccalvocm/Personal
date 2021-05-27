# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:15:14 2021

@author: farrospide
"""
import os
import geopandas as gpd
import contextily as ctx
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


def import_shapes(file):
    folder = os.path.join('..', 'SIG', 
                          'COBERTURAS INFRAESTRUCTURA Y ÁREAS DE RIEGO',
                          'infraest_ariego_etapa2')
    fp = os.path.join(folder, file)
    gdf = gpd.read_file(fp)
    print(gdf.crs)
    return gdf

def import_hidroMAV(file):
    folder = os.path.join('..', 'SIG', 
                          'COBERTURAS INFRAESTRUCTURA Y ÁREAS DE RIEGO',
                          'infraest_ariego_etapa2')
    fp = os.path.join(folder, file)
    gdf = gpd.read_file(fp)
    gdf = gdf.to_crs('EPSG:32719')
    print(gdf.crs)
    return gdf

def import_catchments():
    folder = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH')
    f1 = os.path.join(folder, 'Cuencas', 'Cuencas_DARH_2015_AOHIA_ZC.geojson')
    f2 = os.path.join(folder, 'Subcuencas',
                      'SubCuencas_DARH_2015_AOHIA_ZC.geojson')
    cuencas = gpd.read_file(f1)
    scuencas = gpd.read_file(f2)
    
    return cuencas, scuencas

def import_hidrografia():
    folder = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Hidrografia')
    file = 'RedHidrograficaUTM19S_AOHIA_ZC.geojson'
    fp = os.path.join(folder, file)
    gdf = gpd.read_file(fp)
    print(gdf.crs)
    return gdf

def set_ax(ax, title):
    ax.set_xlabel('Coordenada UTM Este (m)')
    ax.set_ylabel('Coordenada UTM Norte (m)')
    ax.set_title(title)

def plot_gdf(gdf, title, contextshp=None, hgf=None):
    kwargs = {'ec': 'green', 'fc': 'green', 'alpha': 0.7}
    fs = (8.5,11)
    src = ctx.providers.Esri.WorldTerrain
    
    
    fig, ax = plt.subplots(figsize=fs)
    
    if contextshp is not None:
        gdf = gpd.sjoin(gdf, contextshp, how='inner')
        contextshp.plot(ax=ax, fc = 'none', ec = 'red')
    else:
        pass
    
    if hgf is not None:
        hgf = gpd.sjoin(hgf,contextshp,how='inner')
        hgf.plot(ax=ax, color='blue', ls='--', lw = 0.5)
    else:
        pass
    
    
    gdf.plot(ax=ax, **kwargs)
    ctx.add_basemap(ax=ax, zoom=9, source=src, crs='EPSG:32719')
    set_ax(ax, title)
    plt.show()

    
files = ['AR_Maipo_20210430.shp',
         'AR_Rapel_20210414.shp',
         'AR_Mataquito20210507.shp',
         'AR_Maule_20210507.shp']

gdf = import_shapes(files[0])

hidrografiaMAV = import_hidroMAV('Hidro_Maipo_filtro.shp')
hidrografia = import_hidrografia()
cuencas, subcuencas = import_catchments()

maipo = cuencas[cuencas['COD_CUENCA'].isin(['1300'])]

plot_gdf(gdf, 'Areas de Riego\n Cuenca Rio Maipo',contextshp=maipo, hgf=hidrografiaMAV)
    