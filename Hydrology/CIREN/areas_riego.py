# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:15:14 2021

@author: farrospide
"""
import matplotlib
# matplotlib.pyplot.ioff()
import os
import geopandas as gpd
import contextily as ctx
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from unidecode import unidecode

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=height + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

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
    gdf = gpd.read_file(fp, encoding='latin1')
    gdf = gdf[gdf['Dren_Tipo'].isin(['Río', 'Estero', 'Rio', 'RÃ\xado'])]
    print(gdf.crs)
    return gdf

def set_ax(ax, title):
    ax.set_xlabel('Coordenada UTM Este (m)')
    ax.set_ylabel('Coordenada UTM Norte (m)')
    ax.set_title(title)

def plot_gdf(gdf, title, contextshp=None, hgf=None, legarg = None, **kwargs):
    kwargs = {'ec': 'green', 'fc': 'green', 'alpha': 0.7}
    fs = (8.5,11)
    src = ctx.providers.Esri.WorldTerrain
    legend = []
    
    fig, ax = plt.subplots(figsize=fs)
    
    if contextshp is not None:
        gdf = gpd.sjoin(gdf, contextshp, how='inner')
        contextshp.plot(ax=ax, fc = 'none', ec = 'red')
        legend.append(Line2D([0], [0], color='red', lw=4,
                             label='Límite de Cuenca'))
    else:
        pass
    
    if hgf is not None:
        hgf = gpd.sjoin(hgf,contextshp,how='inner')
        hgf.plot(ax=ax, color='blue', ls='--', lw = 0.5)
        legend.append(Line2D([0], [0], color='blue',lw=2,
                             ls='--',label='Cauces principales'))
    else:
        pass
    
    
    gdf.plot(ax=ax, **kwargs)
    if legarg is not None:
        legend.append(legarg)
    else:
        pass
    ctx.add_basemap(ax=ax, zoom=9, source=src, crs='EPSG:32719')
    set_ax(ax, title)
    ax.legend(handles=legend,
              handler_map={mpatches.Circle: HandlerEllipse()},
              loc='best')
    plt.show()
    
def plot_gdf_infra(gdf, title, contextshp=None, hgf=None, ZR=None):
    # boc_kwargs = {'color': 'green', 'marker': 'o', 'markersize': 3}
    can_kwargs = {'color': 'blue', 'ls': '-', 'lw': 0.5}
    emb_kwargs = {'fc': 'cyan', 'ec': 'none'}
    
    
    fs = (8.5,11)
    src = ctx.providers.Esri.WorldTerrain
    legend = []
    
    fig, ax = plt.subplots(figsize=fs)
    
    if contextshp is not None:
        gdf = gpd.sjoin(gdf, contextshp, how='inner')
        contextshp.plot(ax=ax, fc = 'none', ec = 'red')
        legend.append(Line2D([0], [0], color='red', lw=4,
                             label='Límite de Cuenca'))
    else:
        pass
    
    if hgf is not None:
        hgf = gpd.sjoin(hgf,contextshp,how='inner')
        hgf.plot(ax=ax, color='blue', ls='--', lw = 0.5)
        legend.append(Line2D([0], [0], color='blue',lw=2,
                             ls='--',label='Cauces principales'))
    else:
        pass
    
    if ZR is not None:
        ZR = gpd.sjoin(ZR,contextshp,how='inner')
        ZR.plot(ax=ax, fc='brown', alpha = 0.5)
        legend.append(Patch(facecolor='brown', edgecolor='none',
                        label='Áreas de Riego'))
    else:
        pass
    
    # boc = gdf[gdf['tipo'].isin(['Bocatoma'])]
    can = gdf[gdf['tipo'].isin(['Canal'])]
    emb = gdf[gdf['tipo'].isin(['Embalse'])]
    
    # boc.plot(ax=ax, **boc_kwargs)
    can.plot(ax=ax, **can_kwargs)
    emb.plot(ax=ax, **emb_kwargs)
    legend.append(Patch(facecolor='cyan', edgecolor='cyan',
                        label='Embalses y Tranques'))
    legend.append(Line2D([0], [0], color='blue', lw=4,
                             label='Canales'))
    # legend.append(mpatches.Circle((0.5, 0.5),
    #                               radius = 0.25, facecolor = 'green',
    #                               edgecolor = 'none',
    #                               label = 'Bocatomas'))
    ctx.add_basemap(ax=ax, zoom=9, source=src, crs='EPSG:32719')
    set_ax(ax, title)
    ax.legend(handles=legend,
              handler_map={mpatches.Circle: HandlerEllipse()},
              loc='best')
    plt.show()

def savefig(folderpath, filename, ext_img):
    fig = plt.gcf()
    
    filename = os.path.join(folderpath, filename + '.' + ext_img)
    plt.savefig(filename, format = ext_img, bbox_inches = 'tight',
                pad_inches = 0.2)
    plt.close(fig)    
    
#%%

# ---------------------------
# Procedimiento
# ---------------------------


# hidrografiaMAV = import_hidroMAV('Hidro_Maipo_filtro.shp')
hidrografia = import_hidrografia()
cuencas, subcuencas = import_catchments()

    
filesAR = ['AR_Maipo_20210430.shp',
         'AR_Rapel_20210414.shp',
         'AR_Mataquito20210507.shp',
         'AR_CMaule_20210510.shp',
         ]



codcuenca = ['1300', '0600', '0701', '0703']
nomcuenca = ['Río Maipo', 'Río Rapel', 'Río Mataquito', 'Río Maule']

folderpath = os.path.join('..', 'SIG', 
                          'COBERTURAS INFRAESTRUCTURA Y ÁREAS DE RIEGO',
                          'infraest_ariego_etapa2')

legarg = Patch(facecolor='green', edgecolor='none',
                        label='Áreas de Riego')

for AR, cod, nom in zip(filesAR, codcuenca, nomcuenca):
    gdf = import_shapes(AR)
    cca = cuencas[cuencas['COD_CUENCA'].isin([cod])]
    plot_gdf(gdf, 'Áreas de Riego\n Cuenca ' + nom,
             contextshp=cca, hgf=hidrografia, legarg = legarg)
    nom_ext = nom.replace(' ','_')
    nom_ext = unidecode(nom_ext)
    savefig(folderpath, 'AR_' + nom_ext, 'jpg')
    
   
# ----------------------------
# Capa de canales
# ----------------------------

dicfiles = {'Bocatomas_OHiggins.shp': 'Bocatoma',
            'canales_Nuble.shp': 'Canal',
            'Canales_RM.shp': 'Canal',
            'Bocatomas_Maule.shp': 'Bocatoma',
            'Canales_OHiggins.shp': 'Canal',
            'Tranques_CMaipo_20210511.shp': 'Embalse',
            'bocatoma_RM_2019.shp': 'Bocatoma',
            'Tranques_CMataquito_20210511.shp': 'Embalse',
            'Canales_Maule.shp': 'Canal',
            'Tranques_CMaule_20210511.shp': 'Embalse',
            'Tranques_CRapel_20210511.shp': 'Embalse'}


def import_all_shps(folder_path):
    importable = [x for x in os.listdir(folder_path) if (x.endswith('.shp')) and not (x.endswith('20210512.shp'))]
    return importable

path = os.path.join('..', 'SIG',
                    'COBERTURAS INFRAESTRUCTURA Y ÁREAS DE RIEGO',
                    'IR_etapa2')

files = import_all_shps(path)


concat = []
for f in files:
    fp = os.path.join(path,f)
    gdf = gpd.read_file(fp)
    gdf['tipo'] = dicfiles[f]
    concat.append(gdf)
    
gdf = pd.concat(concat, ignore_index=True)

def check_geom(geom):
    try:
        A = geom.geom_type
    except:
        A = 'None'
    return A

for cod, nom, AR in zip(codcuenca, nomcuenca, filesAR):
    cca = cuencas[cuencas['COD_CUENCA'].isin([cod])]
    title = '\n'.join(['Infraestructura de Riego', 'Cuenca ' + nom])
    ZR = import_shapes(AR)
    
    plot_gdf_infra(gdf, title, contextshp=cca)
    savefig(path,nom,'jpg')

# Check geometry type
gdf['Tipo'] = gdf['geometry'].apply(lambda x: check_geom(x))

gdf_joined = gpd.sjoin(gdf,cuencas, how='inner')

tabla = gdf_joined[['NOM_CUENCA','tipo']].value_counts().sort_index()

#%% Plotear capa de embalses como puntos

cuencas, subcuencas = import_catchments()

folder = os.path.join('..', 'SIG',
                    'COBERTURAS INFRAESTRUCTURA Y ÁREAS DE RIEGO',
                    'IR_etapa2')

capas = [x for x in os.listdir(path) if x.endswith('20210512.shp')]

def concat_gdf(folder, shape_list):
    lista = []
    for shape in shape_list:
        path = os.path.join(folder, shape)
        GDF = gpd.read_file(path)
        
        # kwargs_embalses = {'color': 'blue', 'markersize': 4}
        # kwargs_cuencas = {'facecolor': 'none', 'edgecolor': 'red'}
        
        # fig, ax = plt.subplots(figsize = (8.5,11))
        # GDF.plot(ax=ax, **kwargs_embalses)
        # cuencas.plot(ax=ax, **kwargs_cuencas)
        # ctx.add_basemap(ax=ax, zoom = 9,
        #                 source = ctx.providers.Esri.WorldTerrain,
        #                 crs = 'EPSG:32719')
        lista.append(GDF)
    
    GDF_final = pd.concat(lista, ignore_index=True)
    return GDF_final
    
    
gdf_embalses = concat_gdf(folder, capas)

kwargs = {'color': 'blue', 'markersize': 4}
legarg = mpatches.Circle((0.5, 0.5),
                         radius = 0.25, facecolor = 'blue',
                         edgecolor = 'none', label = 'Embalses y tranques')

for cod, nom in zip(codcuenca, nomcuenca):
    gdf = gdf_embalses
    cca = cuencas[cuencas['COD_CUENCA'].isin([cod])]
    plot_gdf(gdf, 'Embalses\n Cuenca ' + nom,
             contextshp=cca, hgf=None, legarg = legarg, **kwargs)
    savefig(folder, 'Embalses_' + nom, 'jpg')
    



