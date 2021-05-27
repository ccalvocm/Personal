#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:17:30 2021

@author: faarrosp
"""

import geopandas as gpd
import os
from unidecode import unidecode
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display
from calendar import isleap
import contextily as ctx

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


def import_subcatchments():
    path_folder= os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH')
    path_macrocuencas =  os.path.join(path_folder, 'Cuencas',
                                      'Cuencas_DARH_2015_AOHIA_ZC.geojson')
    path_subcuencas = os.path.join(path_folder, 'Subcuencas',
                                   'SubCuencas_DARH_2015_AOHIA_ZC.geojson')
    gdfcuencas = gpd.read_file(path_macrocuencas)
    gdfsubcuencas = gpd.read_file(path_subcuencas)
    
    return gdfcuencas, gdfsubcuencas

gdf_cuencas, gdf_subcuencas = import_subcatchments()

# -----------------------
# Limpiar campos y hacer el shp mas ligero, ademas de filtrar por uso
# -----------------------

def limpiar_campos(path, suffix, tipo):
    dic = {'BN': {'USO': ['Bosques', 'Humedales'],
                  'SUBUSO': ['Bosque Nativo', 'Bosque Mixto',
                            'Vegetación Herbácea en Orillas de Ríos',
                            'Vegas',
                            'Otros Terrenos Húmedos',
                            'VegetaciÃ³n HerbÃ¡cea en Orillas de RÃ\xados',
                            'Otros Terrenos HÃºmedos']}}

    gdf = gpd.read_file(path)
    keep = ['USO', 'SUBUSO','CODCOM', 'geometry']
    newgdf = gdf[keep]

    fil1 = newgdf['USO'].isin(dic[tipo]['USO'])
    fil2 = newgdf['SUBUSO'].isin(dic[tipo]['SUBUSO'])

    newnewgdf = newgdf[fil1&fil2]
    newpath = path[:-4] + '_' + suffix + '_light.shp'
    newnewgdf.to_file(newpath)

# --------------------------
# filtrar bosque nativo y humedales + reducir tamanho de archivos
# --------------------------

# for folder, fp in zip(fdpsFORshapes, fpsFORshapes):
#     filepath = os.path.join(fdpFOR, folder, fp)
#     print(filepath)
#     limpiar_campos(filepath, 'BN' ,'BN')


fpsFORshapes = ['Catastro_RV_R05_2013_BN_light.shp',
                'Catastro_RV_R13_2013_BN_light.shp',
                'Catastro_RV_R06_2013_BN_light.shp',
                'Catastro_RV_R07_2016_BN_light.shp']

# paths de sitios RAMSAR, SNASPE y sitios protegidos
fpsitios=os.path.join('..', 'SIG', 'Áreas Protegidas_Shapefile', 'polygon.shp')
fpsnaspe=os.path.join('..', 'SIG', 'Sitios Prioritarios_Shapefile', 'polygon.shp')
fpramsar=os.path.join('..', 'SIG', 'humedales_ramsar', 'humedales_ramsar.shp')


def get_sitios_prioritarios():
    gdf_sitios=gpd.read_file(fpsitios, encoding='latin1') #epsg 32719
    gdf_snaspe=gpd.read_file(fpsnaspe, encoding ='latin1') #epsg 32719
    gdf_ramsar=gpd.read_file(fpramsar, encoding = 'latin1') #epsg 32719 

    gdfcuencas = gpd.read_file(fpcuencas)

    un1 = gpd.overlay(gdf_sitios, gdf_snaspe, how='union')
    un2 = gpd.overlay(un1,gdf_ramsar, how='union')

    intersect = gpd.overlay(un2, gdfcuencas, how='intersection')
    dstfp = os.path.join(fdpFOR, 'sitios_prioritarios.geojson')
    intersect.to_file(dstfp, driver='GeoJSON')

# get_sitios_prioritarios()


def union_forest_layers():
    df_final=[]
    for folder, fp in zip(fdpsFORshapes, fpsFORshapes):
        filepath = os.path.join(fdpFOR, folder, fp)
        df = gpd.read_file(filepath)
        df_final.append(df)

    gdf = pd.concat(df_final, ignore_index=True)

    gdfcuencas = gpd.read_file(fpcuencas)
    intersect = gpd.overlay(gdf,gdfcuencas, how='intersection')

    dstfp = os.path.join(fdpFOR, 'bosque_nativo_y_humedales.geojson')
    intersect.to_file(dstfp, driver='GeoJSON')

# union_forest_layers()


def substract_prioritarios():
    fp = os.path.join(fdpFOR, 'bosque_nativo_y_humedales.geojson')
    bosques = gpd.read_file(fp)
    fp = os.path.join(fdpFOR, 'sitios_prioritarios.geojson')
    prioritarios = gpd.read_file(fp, encoding = 'latin1')
    diff = gpd.overlay(bosques,prioritarios, how='difference')
    dstfp = os.path.join(fdpFOR, 'bosque_nativo_y_humedales_sin_prioritarios.geojson')
    diff.to_file(dstfp,driver='GeoJSON')

# substract_prioritarios()

def plot_shape(path):
    gdf = gpd.read_file(path)

    fig,ax = plt.subplots()
    gdf.plot(ax=ax)

# path = os.path.join(fdpFOR, 'bosque_nativo_y_humedales_sin_prioritarios.geojson')
# plot_shape(path)

# gdf = gpd.read_file(path) # gdf de bosques nativos y humedales sin prioritarios
# gdfscuencas = gpd.read_file(fpsubcuencas)

def get_subcuenca_area(gdftarget, gdfsc, fp):
    gdftarget['for_id'] = gdftarget.index.values
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

# fp = os.path.join(fdpFOR, 'BN_sup_x_scuenca.csv')
# get_subcuenca_area(gdf, gdfscuencas, fp)

def extract_shape_areas_and_simplify():
    path = os.path.join(fdpFOR, 'bosque_nativo_y_humedales_sin_prioritarios.geojson')
    gdf = gpd.read_file(path)
    gdf['for_id'] = gdf.index.values
    gdf['area_m2'] = gdf.area
    gdf.drop(['Shape_Leng', 'Shape_Area', 'Area_km2', 'geometry'],axis=1,inplace=True)
    path = os.path.join(fdpFOR, 'bosque_nativo_y_humedales_sin_prioritarios.csv')
    gdf.to_csv(path)

# extract_shape_areas_and_simplify()

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

def ndays(x):    
    if isleap(x):
        return (1 / 366 / 24 / 60 / 60)
    else:
        return (1 / 365 / 24 / 60 / 60)

def compute_ET():
    path = os.path.join(fdpFOR, 'bosque_nativo_y_humedales_sin_prioritarios.csv')
    gdf = pd.read_csv(path, dtype = {'for_id': int, 'COD_CUENCA': str})

    path = os.path.join(fdpFOR, 'BN_sup_x_scuenca.csv')
    df = pd.read_csv(path, dtype = {'for_id': int, 'COD_DGA': str})

    # df que combina uso, subuso, subcuencas y areas
    gdf = gdf.set_index('for_id').join(df.set_index('for_id'), on = 'for_id')
    gdf['eff_area_m2'] = gdf['area_m2'] * gdf['perc_area']
    
    #  ahora leer el dataframe de precipitaciones por subsubcuenca
    folder_zonalstats = os.path.join('.', 'outputs', 'caracter_hidr', 'cr2MET')
    fp_zonalstats = os.path.join(folder_zonalstats,'CR2MET_estadisticas_x_cuenca.xlsx')
    
    precip = pd.read_excel(fp_zonalstats, dtype = {'COD_CUENCA': str,
                                                   'Ano': int}, 
                           sheet_name = 'SubcuencasDARH')
    precip.rename({'NOM_CUENCA': 'NOM_DGA', 'COD_CUENCA': 'COD_DGA'},
                  axis=1, inplace=True)
    
    precip = precip[['mean','NOM_DGA', 'COD_DGA', 'Ano']]
    
    gdf.reset_index(inplace=True)
    gdf = gdf.join(precip.set_index('COD_DGA'), on = 'COD_DGA')
    
    gdf['ET'] = gdf['mean'].apply(lambda x: ET(x,'Bosques'))
    gdf['Kc'] = 1.0
    
    fil_humedal = gdf['USO'].isin(['Humedales'])
    gdf.loc[fil_humedal,'Kc'] = 1.2
    
    gdf['ET'] = gdf['ET'] * gdf['Kc']
    
    gdf['dda_m3_y'] = gdf['eff_area_m2'] * gdf['ET'] / 1000
    
    gdf['dda_LPS'] = gdf['dda_m3_y'] * gdf['Ano'].apply(lambda x: ndays(x))
    
    #---------------------------
    # Crear pivot table para tener los años por columnas
    #---------------------------        
    pt_sc_lps = gdf.pivot_table(index='COD_DGA', columns = 'Ano',
                                values = 'dda_LPS', aggfunc = 'sum')
    pt_sc_m3y = gdf.pivot_table(index='COD_DGA', columns = 'Ano',
                                values = 'dda_m3_y', aggfunc = 'sum')
    
    # display(gdf.head())
    # display(pt_sc_lps.head())
    # display(pt_sc_m3y.head())
    
    # print(gdf.dtypes)
    # print(precip.dtypes)

    return gdf

def savefig(folderpath, filename, ext_img):
    fig = plt.gcf()
    filename = os.path.join(folderpath, filename + '.' + ext_img)
    plt.savefig(filename, format = ext_img, bbox_inches = 'tight',
                pad_inches = 0.1)
    plt.close(fig)

gdf = compute_ET()


def get_demanda_TS(df, indice, columna):
    df_TS = gdf.pivot_table(columns='Ano', values=columna,index=indice, 
                            aggfunc='sum')
    return df_TS


TS_l_s = get_demanda_TS(gdf, 'COD_CUENCA', 'dda_LPS')

TS_l_s.T.plot(subplots=True, title=['Río Maipo',
                                    'Río Rapel',
                                    'Río Mataquito',
                                    'Río Maule'],
              figsize=(8.5,11), layout=(2,2), legend=False,
              xlabel='Fecha', ylabel='Demanda (L/s)')

plt.suptitle('Demanda Forestal\nUso no productivo')
savefig(fdpFOR, 'Demanda_FOR_no_prod_L_s_macrocuencas', 'jpg')

# ---------------------
# Plotear mapa de demandas
# ---------------------

def get_gdf_geometries(df, gdf, codigo):
    gdfres = gdf.join(df, on=codigo)
    return gdfres

def plot_demand_map(gdf,cuencas,title):
    fig,ax = plt.subplots(figsize=(8.5,11))
    gdf.plot(ax=ax, column=2019, scheme='quantiles', legend=True, cmap='Blues')
    cuencas.plot(ax=ax, fc='none', ec='red', lw=2)
    ctx.add_basemap(ax=ax, crs='EPSG:32719', zoom=9,
                    source=ctx.providers.Esri.WorldTerrain)
    ax.set_xlabel('Coordenada Este UTM (m)')
    ax.set_ylabel('Coordenada Norte UTM (m)')
    ax.set_title('Demanda instantánea año 2019\n' + title)
    
    

TS_subcuencas_l_s = get_demanda_TS(gdf, 'COD_DGA', 'dda_LPS')
gdf_subcuencas_L_s_no_prod = get_gdf_geometries(TS_subcuencas_l_s, gdf_subcuencas, 'COD_DGA')

plot_demand_map(gdf_subcuencas_L_s_no_prod, gdf_cuencas, 'No productivo')
savefig(fdpFOR, 'Demanda_FOR_no_prod_subsubcuencas', 'jpg')


fp = os.path.join(fdpFOR, 'Demanda_FOR_no_prod_TS_macrocuencas.xlsx')
TS_l_s.to_excel(fp)


# plotear demanda productiva
fp = os.path.join(fdpFOR, 'dda_FOR_plant.xlsx')
df = pd.read_excel(fp, sheet_name= 'cuenca_lps')
df.set_index('NOM_CUENCA', inplace=True)
df.reindex(['Río Maipo',
            'Río Rapel',
            'Río Mataquito y afluentes',
            'Río Maule']).T.plot(subplots=True, title=['Río Maipo',
                                    'Río Rapel',
                                    'Río Mataquito',
                                    'Río Maule'],
              figsize=(8.5,11), layout=(2,2), legend=False,
              xlabel='Fecha', ylabel='Demanda (L/s)')
plt.suptitle('Demanda Forestal\nUso productivo')
savefig(fdpFOR, 'Demanda_FOR_prod_L_s_macrocuencas', 'jpg')

TS_subcuencas_prod_l_s = pd.read_excel(fp, sheet_name= 'lps',dtype={'COD_DGA': int}, index_col=0)
gdf_subcuencas['COD_DGA'] = gdf_subcuencas['COD_DGA'].astype(int)
gdf_subcuencas_L_s_prod = get_gdf_geometries(TS_subcuencas_prod_l_s, gdf_subcuencas, 'COD_DGA')
plot_demand_map(gdf_subcuencas_L_s_prod, gdf_cuencas, 'Uso productivo')
savefig(fdpFOR, 'Demanda_FOR_prod_subsubcuencas', 'jpg')

# Obtener suma de usos
df.index.name='COD_CUENCA'
df.index = ['1300', '0701', '0703', '0600']
TS_l_s.index = ['1300', '0600', '0701', '0703']
TS_l_s.columns = [int(x) for x in TS_l_s.columns]
sum_df = pd.concat([df, TS_l_s])
sum_df = sum_df.groupby(level=0).sum()
sum_df.reindex(['1300', '0600', '0701', '0703']).T.plot(subplots=True, title=['Río Maipo',
                                    'Río Rapel',
                                    'Río Mataquito',
                                    'Río Maule'],
              figsize=(8.5,11), layout=(2,2), legend=False,
              xlabel='Fecha', ylabel='Demanda (L/s)')
plt.suptitle('Demanda Forestal\nTotal')
savefig(fdpFOR, 'Demanda_FOR_total_TS_macrocuencas', 'jpg')

fp = os.path.join(fdpFOR, 'Demanda_FOR_total_TS_macrocuencas.xlsx')
sum_df.T.to_excel(fp)



