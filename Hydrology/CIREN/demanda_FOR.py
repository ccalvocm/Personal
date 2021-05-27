#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:17:34 2021

@author: faarrosp
"""

import os
import geopandas as gpd
import rasterio as rio
import pandas as pd
from unidecode import unidecode
from matplotlib import pyplot as plt
import contextily as ctx


def get_intersection(df1, df2):
    df = gpd.overlay(df1,df2,how='intersection')
    return df

# ----------------------
# Paths
# ----------------------

# paths de shapes de catastros CONAF
fdpFOR = os.path.join('..','Etapa 1 y 2', 'Demanda', 'FOR')
fdpsFORshapes = ['V', 'RM', 'VI', 'VII']
fpsFORshapes = ['Catastro_RV_R05_2013.shp',
                'Catastro_RV_R13_2013.shp',
                'Catastro_RV_R06_2013.shp',
                'Catastro_RV_R07_2016.shp']

# paths de shapes de cuencas y subcuencas
fdpcuencas = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH')
fpcuencas = os.path.join(fdpcuencas, 'Cuencas',
                         'Cuencas_DARH_2015_AOHIA_ZC.geojson')

fpsubcuencas = os.path.join(fdpcuencas, 'Subcuencas',
                            'SubCuencas_DARH_2015_AOHIA_ZC.geojson')


# paths de sitios RAMSAR, SNASPE y sitios protegidos
fpsitios=os.path.join('..', 'SIG', 'Áreas Protegidas_Shapefile', 'polygon.shp')
fpsnaspe=os.path.join('..', 'SIG', 'Sitios Prioritarios_Shapefile', 'polygon.shp')
fpramsar=os.path.join('..', 'SIG', 'humedales_ramsar', 'humedales_ramsar.shp')

#-----------------------
# Importar los shapes de cuencas y subcuencas
# ----------------------
gdfcuencas = gpd.read_file(fpcuencas)
gdfsubcuencas = gpd.read_file(fpsubcuencas)

# ----------------------
# Importar los shapes de las coberturas forestales CONAF
# ----------------------

gdfs = []

for folder, path in zip(fdpsFORshapes[:1], fpsFORshapes[:1]):
    fp = os.path.join(fdpFOR, folder, path)
    gdf = gpd.read_file(fp)
    gdf_intersection = get_intersection(gdf, gdfcuencas)
    gdfs.append(gdf_intersection)

# concatenar todas las coberturas en una sola
gdf_final = pd.concat(gdfs, ignore_index=True)

# ----------------------
# Importar el shape filtrado de bosque plantacion
# ----------------------
    
# fpbosqueplantacion = os.path.join(fdpFOR, 'AOHIA_ZC_bosqes_plantacion.geojson')
# gdfbosqueplantacion = gpd.read_file(fpbosqueplantacion)

# fpnoprod = os.path.join(fdpFOR, 'AOHIA_ZC_no_productivo.geojson')
# gdfnoprod = gpd.read_file(fpnoprod)



gdf_sitios=gpd.read_file(fpsitios) #epsg 32719
gdf_snaspe=gpd.read_file(fpsnaspe) #epsg 32719
gdf_ramsar=gpd.read_file(fpramsar) #epsg 32719

#%% Trabajar con el shape bruto

#--------------------------------
# obtener la diferencia de bosques con sitios protegidos
#--------------------------------

# 1. unir las capas de sitios prioritarios, snaspe y ramsar
union = gpd.overlay(gdf_sitios,gdf_snaspe, how='union')
union = gpd.overlay(union, gdf_ramsar, how = 'union')

# 2. dejar solo lo que esta adentro de las cuencas de estudio
gdf = gpd.overlay(gdf, gdfcuencas, how = 'intersection')
union = gpd.overlay(union, gdfcuencas, how = 'intersection')

# 3. restar a los bosques nativos y humedales los sitios protegidos
diff = gpd.overlay(gdf, union, how='difference')

#%%

#%%

# ----------------------
# Filtrar por uso bosque plantacion
# ----------------------

# gdf.columns = [unidecode(x) for x in gdf.columns]

# fil1 = gdf['USO'].isin(['Bosques'])
# fil2 = gdf['SUBUSO'].isin(['PlantaciÃ³n', 'Plantación])

# gdf = gdf[fil1 & fil2]

# ----------------------
# Filtrar por uso bosque nativo y humedales
# ----------------------

# gdf.columns = [unidecode(x) for x in gdf.columns]

# fil1 = gdf['USO'].isin(['Bosques', 'Humedales'])
# fil2 = gdf['SUBUSO'].isin(['Bosque Nativo', 'Bosque Mixto',
#                            'Vegetación Herbácea en Orillas de Ríos',
#                            'Vegas',
#                            'Otros Terrenos Húmedos',
#                            'VegetaciÃ³n HerbÃ¡cea en Orillas de RÃ\xados',
#                            'Otros Terrenos HÃºmedos',
#                            ''])

# gdf = gdf[fil1 & fil2]
# ----------------------
# Filtrar dentro de las cuencas en estudio
# ----------------------

# gdf2 = gpd.sjoin(gdf,gdfcuencas, how='inner')

# fpsave = os.path.join(fdpFOR, 'AOHIA_ZC_bosqes_plantacion.geojson')
fpsave = os.path.join(fdpFOR, 'AOHIA_ZC_no_productivo.geojson')
diff.to_file(fpsave, driver = 'GeoJSON')

#%% 

# gdfbosqueplantacion['for_id'] = gdfbosqueplantacion.index.values
# diff['for_id'] = diff.index.values
gdfnoprod['for_id'] = gdfnoprod.index.values
# Calcular la porción (porcentaje) que una subcuenca es de una comuna, 
# para luego multiplicar este porcentaje por el consumo y asi obtener un
# consumo por subcuenca

# fp = os.path.join(fdpFOR, 'porcentaje_bosqueplantacion_subcuenca.csv')
fp = os.path.join(fdpFOR, 'porcentaje_noprod_subcuenca.csv')
#-------------------------
# Crear archivo de cobertura de bosque por subcuenca
#-------------------------

def get_subcuenca_area(gdftarget, gdfsc):
    with open(fp, 'w') as file:
        file.write('for_id,COD_DGA,perc_area\n')
        for idcod, forma in zip(gdftarget['for_id'], gdftarget['geometry']):
            for codsubc, subcuenc in zip(gdfsc['COD_DGA'], gdfsc['geometry']):
                try:
                    if subcuenc.intersects(forma):
                        perc_inter = subcuenc.intersection(forma).area/forma.area
                        #print(codsubc)
                        file.write(','.join([str(idcod), str(codsubc), str(perc_inter) + '\n']))
                except:
                        pass

# get_subcuenca_area(diff, gdfsubcuencas)
#%%

#----------------------------
# ahora debemos asignar un valor de ET a cada una de estas features
# para ello, importamos la planilla de valores de P segun CR2MET
#----------------------------

dfareas = pd.read_csv(fp)

fdpcr2zonalstats = os.path.join('.', 'outputs', 'caracter_hidr', 'cr2MET')
fpcr2zonalstats = os.path.join(fdpcr2zonalstats,'CR2MET_estadisticas_x_cuenca.xlsx')

dfcr2 = pd.read_excel(fpcr2zonalstats, sheet_name = 'SubcuencasDARH')

#-----------------------------
# aca multiplicamos el area de cada poligono de bosque por porcentaje de
# que abarca cada subcuenca sobre esta. Ahora ya no necesitamos mas este
# dataframe, salvo pivotear para sumar areas sobre todas las subcuencas
#-----------------------------
gdfareas = gdfnoprod.copy()
gdfareas = gdfareas.join(dfareas.set_index('for_id'), on='for_id')
gdfareas['eff_area'] = gdfareas.area * gdfareas['perc_area']



#-----------------------------
# ahora obtenemos la evapotranspiracion por subcuenca por anho
#-----------------------------

def ET(row, tipo):
    '''
    

    Parameters
    ----------
    row : float
        precipitacion anual en mm/ano.
    tipo : string
        tipo de cobertura vegetal a utilizar; determina el coeficiente f.

    Returns
    -------
    et : float
        evapotranspiracion en mm/ano segun formula de Zhang.

    '''
    dic = {'plantacion':0.75,
           'Bosques': 0.0,
           'Humedales': 0.0}
    f = dic[tipo]
    P = row
    a1, a2 = 1410/P, 1100/P
    et = ((1+2*a1)/(1+2*a1+a1**(-1))*f + (1-f)*(1+0.5*a2)/(1+0.5*a2+a2**(-1)))* P

    return et

filbosque = gdfareas['USO'].isin(['Bosques'])
filhumedal = gdfareas['USO'].isin(['Humedales'])

dfcr2['ET'] = 0.0

dfcr2['ET'] = dfcr2['mean'].apply(lambda x: ET(x,'Bosques'))






#---------------------------
# queremos ahora extraer solo las del anho 2015
#---------------------------

# dfcr2 = dfcr2[dfcr2['Ano'].isin([2015])]
dfcr2.rename({'COD_CUENCA': 'COD_DGA'}, axis=1, inplace=True)

#---------------------------
# asignemos ahora la evapotranspiracion a cada subsubcuenca
fields = ['COD_DGA', 'Ano', 'ET', 'NOM_CUENCA']
df_areas_x_subcuenca = gdfareas.join(dfcr2[fields].set_index('COD_DGA'), on='COD_DGA')


#---------------------------
# ahora multiplicamos la evapotranspiracion por la superficie x subcuenca
# para obtener el volumen:
# demanda [m3/año] = ET (mm/año) * eff_area (m2) 

df_areas_x_subcuenca['dda_m3_y'] = df_areas_x_subcuenca['ET'] * \
    df_areas_x_subcuenca['eff_area'] / 1000

from calendar import isleap    
#---------------------------
# expresemos la demanda en L/s
#---------------------------
def ndays(x):    
    if isleap(x):
        return (1 / 366 / 24 / 60 / 60)
    else:
        return (1 / 365 / 24 / 60 / 60)
    
df_areas_x_subcuenca['dda_LPS'] = df_areas_x_subcuenca['dda_m3_y'] * \
    df_areas_x_subcuenca['Ano'].apply(lambda x: ndays(x))


#---------------------------
# Crear pivot table para tener los años por columnas
#---------------------------        
pt_sc_lps = df_areas_x_subcuenca.pivot_table(index='COD_DGA', columns = 'Ano', values = 'dda_LPS')
pt_sc_m3y = df_areas_x_subcuenca.pivot_table(index='COD_DGA', columns = 'Ano', values = 'dda_m3_y')

#---------------------------
# ahora crucemos con el geodataframe de subcuencas para plotear
#---------------------------
gdfsubcuencas.set_index('COD_DGA', inplace=True)
gdfsubcuencas.index = gdfsubcuencas.index.astype(int)
gdf_sc_lps = gdfsubcuencas.join(pt_sc_lps, on='COD_DGA')
gdf_sc_m3y = gdfsubcuencas.join(pt_sc_m3y, on='COD_DGA')


fig, ax = plt.subplots(figsize=(8.5,11))
# gdf_sc_lps.plot(ax=ax, column = 2015, scheme='quantiles',legend=True)
gdf_sc_m3y.plot(ax=ax, column = 2015, scheme='quantiles',legend=True, ec='black')
gdfcuencas.plot(ax=ax, fc='none', ec='black', ls='--')
ctx.add_basemap(ax=ax, source=ctx.providers.Esri.WorldTerrain, crs='EPSG:32719')
ax.set_title('Demanda FORESTAL\nPlantacion Forestal')
ax.set_xlabel('Coordenada UTM Este (m)')
ax.set_ylabel('Coordenada UTM Norte (m)')
plt.show()

#-------------------------------
# export to excel
#-------------------------------
fields2export = [x for x in range(1979,2021)]
fields2export.append('NOM_CUENCA')

df1 = gdf_sc_lps[fields2export].groupby('NOM_CUENCA').sum()
df2 = gdf_sc_m3y[fields2export].groupby('NOM_CUENCA').sum()
dstfp = os.path.join(fdpFOR, 'dda_FOR_plant.xlsx')
with pd.ExcelWriter(dstfp) as writer:  
    pt_sc_lps.to_excel(writer, sheet_name='lps')
    pt_sc_m3y.to_excel(writer, sheet_name='m3y')
    df1.to_excel(writer, sheet_name='cuenca_lps')
    df2.to_excel(writer, sheet_name='cuenca_m3y')
    
