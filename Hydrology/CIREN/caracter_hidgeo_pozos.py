#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 15:52:43 2021

@author: faarrosp
"""
# import libraries
import matplotlib
# matplotlib.pyplot.ioff()
import pandas as pd
import geopandas as gpd
import os
import contextily as ctx
from matplotlib import pyplot as plt
from unidecode import unidecode
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import rcParams, cycler

# ---------------------------- plotting options
fsize = (22,17) # (width, height)
scmap = 'terrain' #subcatchment colormap
aqcmap = 'magma' #aquifer colormap
scalpha = 0.4 # subcatchment alpha
aqalpha = 0.2 # aquifer alpha
crs = 'EPSG:32719' # crs
provider = ctx.providers.Esri.WorldTerrain
ctxzoom=9
# cmap = plt.cm.coolwarm



cuencakwgs = {'fc': 'none', 'ec' : 'red', 'ls': '--', 'lw': 4}
pozoskwgs = {'color': 'blue', 'marker': '^'}
hidrokwgs = {'color':'teal', 'linewidth': 1.0, 'linestyle': '--'}
geolkwgs = {'cmap': 'magma', 'alpha': 0.2}
aquifkwgs = {'cmap': 'magma', 'alpha': 0.2, 'ec': 'black'}
legend = [Line2D([0], [0], color='teal', lw=4, label='Hidrografía'),
          Line2D([0], [0], marker='^', color='w', label='Pozos DGA',
                 markerfacecolor='blue', markersize=15),
          Line2D([0], [0], color='red',lw=4,ls='--',label='Límite de cuencas'),
          Patch(facecolor='orange', edgecolor='r', label='Acuíferos DGA')]

#%%
# ---------------------------- define paths

# path de geojson de cuencas
cuencasfdp = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH')
cuencasfp = os.path.join(cuencasfdp, 'Cuencas', 'Cuencas_DARH_2015_AOHIA_ZC.geojson')
cuencas = gpd.read_file(cuencasfp)

# path de excel de pozos
pozosfdp = os.path.join('..', 'Etapa 1 y 2', 'Aguas subterráneas')
pozosfp = os.path.join(pozosfdp, 'Pozos_DGA_CFA.xlsx')
pozos = pd.read_excel(pozosfp, sheet_name = 'BNAT_Niveles_Poz',
                      converters = {'FECHA': lambda x: pd.to_datetime(x,dayfirst=True)})

# path de hidrografia
hidrofdp = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Hidrografia')
hidrofp = os.path.join(hidrofdp, 'RedHidrograficaUTM19S_AOHIA_ZC.geojson')
hidrografia = gpd.read_file(hidrofp)

# path de mapoteca DGA (formaciones geologicas y acuiferos)
dgafdp = os.path.join('..', 'SIG', 'Mapoteca DGA', 'Mapoteca_DGA', '02_DGA')
dgaaquiferfp = os.path.join(dgafdp, 'Acuiferos', 'ACUIFEROS_SHAC_Sep_2020.shp')
dgageologiafp = os.path.join(dgafdp, 'Publicacion_Mapa_Hidrogeologico',
                             'ocurrencia_aguas_subterraneas.shp')
acuiferos = gpd.read_file(dgaaquiferfp)
geologia = gpd.read_file(dgageologiafp)

savefdp = os.path.join('.', 'outputs', 'caracter_hidr', 'aguassubterraneas')

#%%
def savefig(folderpath, filename, ext_img):
    fig = plt.gcf()
    filename = os.path.join(folderpath, filename + '.' + ext_img)
    plt.savefig(filename, format = ext_img, bbox_inches = 'tight',
                pad_inches = 0.1)
    plt.close(fig)

def filter_pozos_by_gdf(df,gdf):
    df = df[['CODIGO', 'NOMBRE', 'UTM NORTE', 'UTM ESTE']].drop_duplicates().sort_values(by = 'NOMBRE')
    outgdf = gpd.GeoDataFrame(df, crs = 'EPSG:32719',
                          geometry = gpd.points_from_xy(df['UTM ESTE'], df['UTM NORTE']))
    outgdf = sjoin(outgdf, gdf, how = 'inner')
    return outgdf

def set_map_labels_n_title(cax, titulo):
    cax.set_xlabel('Coordenada UTM Este (m)')
    cax.set_ylabel('Coordenada UTM Norte (m)')
    cax.set_title(titulo)
    
def set_ts_labels_n_title(cax, titulo):
    cax.set_xlabel('Fecha')
    cax.set_ylabel('Profundidad nivel pozo (m)')
    cax.set_title(titulo)    
    
def create_gdf_from_geom(geometry, crs):
    gseries = gpd.GeoSeries(geometry, crs = crs)
    gdf = gpd.GeoDataFrame(gseries)
    gdf.rename({0: 'geometry'}, axis = 1, inplace=True)
    gdf.crs = crs
    return gdf

def drop_indices(df):
    try:
        df.drop('index_left', axis=1, inplace=True)
    except:
        try:
            df.drop('index_right', axis=1, inplace=True)
        except:
            pass
    return df

def sjoin(ldf, rdf, how):
    ldf = drop_indices(ldf)
    rdf = drop_indices(rdf)
    df = gpd.sjoin(ldf,rdf, how=how)
    return df

def create_TS(timex,y):
    start = min(timex)
    end = max(timex)
    rng = pd.date_range(start,end, freq='D')
    skeleton = pd.DataFrame(index=rng)
    olddf = pd.DataFrame(y)
    olddf.set_index(timex,inplace=True)
    TS = skeleton.merge(olddf, left_index=True, right_index=True, how='left')
    return TS

#%%
    
# ------------------- Plotear pozos en las macrocuencas
# filtra los pozos por las macrocuencas
sns.set()
sns.set_style("ticks")

gdfaquif = sjoin(acuiferos, cuencas, how = 'inner')
gdfpozos = filter_pozos_by_gdf(pozos, gdfaquif)

# fig, ax = plt.subplots(figsize = fsize)

# gdfaquif.plot(ax=ax, **aquifkwgs)
# cuencas.plot(ax=ax, **cuencakwgs)
# ctx.add_basemap(ax=ax, zoom=ctxzoom, crs=crs, source=provider)
# hidrografia.plot(ax=ax, **hidrokwgs)
# gdfpozos.plot(ax=ax, **pozoskwgs)

# ax.legend(handles=legend, loc='best')
# set_map_labels_n_title(ax, 'Pozos de Monitoreo DGA\nCuencas en estudio')
# savefig(savefdp, 'pozos_dga_Macrocuencas', 'jpg')


# # ------------------- Plotear pozos en cada macrocuenca
# for ix, row in cuencas.iterrows():
#     cca = create_gdf_from_geom(row['geometry'], crs = 'EPSG:32719')
#     name = row['NOM_CUENCA']
#     gdfaquif = sjoin(acuiferos, cca, how='inner')
#     gdfpozos = filter_pozos_by_gdf(pozos, gdfaquif)
#     gdfhidro = sjoin(hidrografia, cca, how='inner')
    
#     fig,ax = plt.subplots(figsize = fsize)
    
#     gdfaquif.plot(ax=ax, **aquifkwgs)
#     cca.plot(ax=ax, **cuencakwgs)
#     ctx.add_basemap(ax=ax, zoom=ctxzoom, crs=crs, source=provider)
#     gdfhidro.plot(ax=ax, **hidrokwgs)
#     gdfpozos.plot(ax=ax, **pozoskwgs)
    
#     ax.legend(handles=legend, loc='best')
#     set_map_labels_n_title(ax, 'Pozos de Monitoreo DGA\nCuencas ' + name)
#     savefig(savefdp, 'pozos_dga_' + name, 'jpg')
    

# %% 
# ------------------ Plotear series de tiempo por pozo

# sns.set()
# sns.set_style("ticks")

for ix, row in cuencas.iterrows():
    cca = create_gdf_from_geom(row['geometry'], crs = 'EPSG:32719')
    name = row['NOM_CUENCA']
    gdfaquif = sjoin(acuiferos, cca, how='inner')
    gdfpozos = filter_pozos_by_gdf(pozos, gdfaquif)
    
    for ix2, row2 in gdfpozos.iterrows():
        cod = row2['CODIGO']
        nam = row2['NOMBRE']
        nam2 = unidecode(row2['NOMBRE']).replace('/','_')
        nam2 = nam2.replace("'", '')
        nam2 = nam2.replace('"', '')
        nam2 = nam2.replace('.', ' ')
        nam2 = nam2.replace(' ', '_')
        
        
        fig, ax = plt.subplots(figsize = (8.5,11))
        pozos[pozos['CODIGO'].isin([cod])].plot(x='FECHA', y='Valor', ax=ax,
                                                legend=False)
        # x = pozos[pozos['CODIGO'].isin([cod])]['FECHA']
        # y = pozos[pozos['CODIGO'].isin([cod])]['Valor']
        # ts = create_TS(x, y)
        # ts.plot(marker = 'o')
        plt.gca().invert_yaxis()
        plt.gca().grid()
       
        set_ts_labels_n_title(ax, '\n'.join(['Hidrograma de profundidad de nivel de pozo',
                                              'Pozo '+ nam,
                                              'Código ' + cod]))
        name = unidecode(name)
        name = name.replace(' ','_')
        nam2 = unidecode(nam2)
        nam2 = nam2.replace(' ', '')
        savefig(savefdp, '_'.join(['TS', name, cod, nam2]), 'pdf')


#%% Incluir todas las TS en un pdf anexo

filelist = [x for x in os.listdir(savefdp) if ('TS' in x and x.endswith('.pdf'))]
filelist.sort()

with open('Anexo_Pozos_TS.tex', 'w+') as f:
    for file in filelist:
        text = '\\includepdf[pages=-]{' + file +'}'
        f.write(text + '\n')
