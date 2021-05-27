# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:26:53 2020

@author: farrospide
"""
import matplotlib
from matplotlib import pyplot as plt
import contextily as ctx
import pandas as pd
import dataframe_image as dfi
from PIL import Image
import seaborn as sns
import datetime
import geopandas

def plot_catchment_map(bsn_df, bsn_N, ax, basemap, **kwargs):
    """Plots the shape of the catchment and a basemap underneath it, given the catchment DARH code

    Parameters
    ----------
    bsn_df : GeoDataFrame (from GeoPandas module)
        Dataframe of polygons containing catchment geometries
    bsn_N : str
        DARH catchment code e.g: '1300' for Rio Maipo
    basemap: bool
        set to False as default. True if you wish to have a basemap plotted underneath the catchment
    ax: Matplotlib Axes object

    Returns
    -------
    p: the plotted ax object
        
    """
    
    filt = bsn_df['COD_CUENCA'] == bsn_N
    basin = bsn_df.loc[filt]
    p = basin.plot(ax = ax, alpha=0.5, **kwargs)
    
    if basemap:
        ctx.add_basemap(ax = ax, crs= basin.crs.to_string(),
                        source = ctx.providers.Esri.WorldTerrain, zoom = 8)
        x, y, arrow_length = 0.95, 0.95, 0.07
        ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15),
                    ha='center', va='center', fontsize=20,
                    xycoords=ax.transAxes)
        x, y, scale_len = basin.bounds['minx'], basin.bounds['miny'].min(), 20000 #arrowstyle='-'
        scale_rect = matplotlib.patches.Rectangle((x,y),scale_len,200,linewidth=1,
                                                edgecolor='k',facecolor='k')
        ax.add_patch(scale_rect)
        plt.text(x+scale_len/2, y+5000, s='20 KM', fontsize=10,
                 horizontalalignment='center')
    else:
        pass

    return p
    
def plot_gdf(gdf, ax, basemap, **kwargs):
    p = gdf.plot(ax = ax, alpha=0.5, **kwargs)
    if basemap:
        ctx.add_basemap(ax = ax, crs= gdf.crs.to_string(),
                        source = ctx.providers.Esri.WorldTerrain, zoom = 8)
    else:
        pass
    
    return p

    
def plot_catchment_table(bsn_df, bsn_N, ax, path, fields, **dfi_kwargs):
    """Plots the contents of a DARH shp attribute table, given the catchment DARH code

    Parameters
    ----------
    bsn_df : GeoDataFrame (from GeoPandas module)
        Dataframe of polygons containing catchment geometries
    bsn_N : str
        DARH catchment code e.g: '1300' for Rio Maipo
    ax: matplotlib Axes object
        axes where to put the table
    path: str
        path of the exported table will be saved as image (must include extension (PNG, PDF, etc))
    fields: list of str
        attributes to be exported to the table
    **dfi_kwargs: kwargs of the dataframe_image library


    Returns
    -------
    p: the plotted ax object
        
    """
    bsn_df = bsn_df.loc[bsn_df['COD_CUENCA'] == bsn_N]
    bsn_df = bsn_df[fields]
    bsn_df['Area_Km2'] = round(bsn_df['Area_Km2'],0)
    bsn_df.reset_index(drop = True, inplace = True)
    bsn_df.style.hide_index()
    dfi.export(bsn_df, path, **dfi_kwargs)
    img = Image.open(path)
    ax.imshow(img)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
def plot_attr_value(bsn_df, bsn_N, ax, field, **kwargs):
    '''
    

    Parameters
    ----------
    bsn_df : TYPE
        DESCRIPTION.
    bsn_N : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.
    field : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    bsn_df = bsn_df.loc[bsn_df['COD_CUENCA'] == bsn_N]
    bsn_df.reset_index(drop = True, inplace = True)
    if field == 'index':
        bsn_df['N'] = bsn_df.index
        field = 'N'
    else:
        pass
    bsn_df.apply(lambda x: ax.annotate(text=x[field], xy=x.geometry.centroid.coords[0], ha='center', **kwargs),axis=1)

def plot_diagrama_cruces(df, yini, yfin, ax):
    '''
    

    Parameters
    ----------
    df : DataFrame
        Dataframe de informacion fluviometrica.
    yini : int
        año de inicio de filtrado.
    yfin : int
        año final de filtrado.
    ax : Matplotlib Axes object
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    yini = datetime.datetime.strptime("01-01-" + str(yini), "%m-%d-%Y")
    yfin = datetime.datetime.strptime("12-31-" + str(yfin), "%m-%d-%Y")
    df_rs = df.loc[yini:yfin]
    df_rs = df_rs.resample('Y').count()/365*100
    df_rs.index = df_rs.index.year
    df_rs.index.names = ['Año']    
    cmap = 'tab20c_r' #greys, 
    sns.heatmap(df_rs.T, ax = ax, cmap = cmap, linewidths = .5,
                cbar_kws={'label': '% data dispon.'})#,
                        # yticklabels = nombres)
        # plt.ylabel('Cod. Estacion')
        # plt.xlabel('Año')
        # plt.title('Data diaria ' + vartit[variable] + '\n' + titulos[cuenca])
    plt.yticks(rotation=0) 

def plot_dataframe_table(df, ax, path, fields, **dfi_kwargs):
    """Plots the contents of a DARH shp attribute table, given the catchment DARH code

    Parameters
    ----------
    df : Pandas DataFrame 
        Dataframe
    ax: matplotlib Axes object
        axes where to put the table
    path: str
        path of the exported table will be saved as image (must include extension (PNG, PDF, etc))
    fields: list of str
        attributes to be exported to the table
    **dfi_kwargs: kwargs of the dataframe_image library


    Returns
    -------
    p: the plotted ax object
        
    """
    # bsn_df = bsn_df.loc[bsn_df['COD_CUENCA'] == bsn_N]
    # bsn_df = bsn_df[fields]
    # bsn_df['Area_Km2'] = round(bsn_df['Area_Km2'],0)
    # bsn_df.reset_index(drop = True, inplace = True)
    # bsn_df.style.hide_index()
    dfi.export(df[fields], path, **dfi_kwargs)
    img = Image.open(path)
    ax.imshow(img)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def DGA_to_gdf(df, CRS):
    gdf = geopandas.GeoDataFrame(df, crs = CRS,
                                 geometry = geopandas.points_from_xy(df['UTM Este'], df['UTM Norte']))
    return gdf
