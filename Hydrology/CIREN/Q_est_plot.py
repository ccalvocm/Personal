# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 13:18:36 2021

@author: Carlos
"""

import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import modules_FAA
from unidecode import unidecode
import modules_CCC
import contextily as ctx
from matplotlib.lines import Line2D
import matplotlib
from matplotlib.patches import Patch
import locale
locale.setlocale(locale.LC_NUMERIC, "es_ES")

# --------------------------------------------------
# preámbulo
# ---------------------------------------------

cuenca = 'Mataquito'
ruta_OD = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos\datosDGA\Q'


dicc_ruta = {'Maipo' : ['\\Maipo\\RIO MAIPO_Q_mensual.xlsx','1300','../Etapa 1 y 2/datos/datosDGA/Q/Maipo/Maipo_cr2corregido_Q.xlsx'], 
             'Rapel' : ['\\Rapel\\RIO RAPEL_Q_mensual.xlsx','0600','../Etapa 1 y 2/datos/datosDGA/Q/Rapel/RIO RAPEL_Q_mensual.xlsx'], 
             'Mataquito' : ['\\Mataquito\\RIO MATAQUITO_Q_mensual.xlsx','0701','../Etapa 1 y 2/datos/datosDGA/Q/Mataquito/RIO MATAQUITO_mensual.xlsx'],
             'Maule' : [r'../Etapa 1 y 2/datos/RIO MAULE_mensual.xlsx','0703',r'../Etapa 1 y 2/datos/RIO MAULE_mensual.xlsx'],
             'Nuble' : [r'../Etapa 1 y 2/datos/RIO ÑUBLE_Q_mensual.xlsx','0801',r'../Etapa 1 y 2/datos/RIO ÑUBLE_Q_mensual.xlsx']}

def plotsEst(cuenca):
    
    #rutas
    # --------------------------------------------------
    path_excel =  dicc_ruta[cuenca][2]
    path = '../Etapa 1 y 2/GIS/Cuencas_DARH/Cuencas/Cuencas_DARH_2015.shp'
    ruta_hidrograph = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\GIS\Hidrografia\Red_Hidrografica.shp'
    ruta_cuencas = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\SIG\Mapoteca DGA\Mapoteca_DGA\02_DGA\Cuencas\Cuencas_DARH_2015_Cuencas.shp'
    # ----------------------------------------------------
    
    # cuenca
    cuenca_shp = geopandas.read_file(ruta_cuencas)    
    cuenca_shp = cuenca_shp[cuenca_shp['COD_CUENCA'] == dicc_ruta[cuenca][1]]
    
    # hidrografia
    hidrograph = geopandas.read_file(ruta_hidrograph)
    hidrograph = hidrograph.to_crs(epsg = 32719)
    # clip y recuento de hidrografía
    gdf_hidrograf = geopandas.clip(hidrograph, cuenca_shp)
        
    est_q = pd.read_excel(path_excel, sheet_name = 'info estacion')
        
    if cuenca in ['Mataquito', 'Maule']:
        gdf =  geopandas.GeoDataFrame(est_q, geometry=geopandas.points_from_xy(est_q['Lon'], est_q['Lat']))
        gdf.set_crs(epsg=4326 , inplace=True)
        gdf.to_crs(epsg=32719 , inplace=True)
    else:
        gdf =  geopandas.GeoDataFrame(est_q, geometry=geopandas.points_from_xy(est_q['UTM Este'], est_q['UTM Norte']))
        gdf.set_crs(epsg=32719, inplace=True)
    gdf.to_file(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\QAQC\Q\gdf'+cuenca+'.shp')
    
    basin = geopandas.read_file(path)
        
    fig = plt.figure(figsize=(11, 8.5))
    axes = fig.add_subplot(111)
    
    gdf.plot(ax = axes, color = 'cyan', zorder=3)
    # gdf_hidrograf = geopandas.read_file(ruta_hidrograf)
    gdf_hidrograf.plot(ax = axes)
    modules_FAA.plot_catchment_map(basin, bsn_N = dicc_ruta[cuenca][1], ax = axes, basemap = True, linewidth = 5, edgecolor = 'black', label = 'Limite cuenca')
    axes.set_xlabel('Coordenada Este WGS84 19S (m)')
    axes.set_ylabel('Coordenada Norte WGS84 19S (m)')
    axes.legend(['Cuenca Río '+cuenca,'Estaciones fluviométricas utilizadas','Hidrografía'], loc = 'upper left')

def xDiagram(cuenca):
    path =  dicc_ruta[cuenca][2]
    df = pd.read_excel(path, sheet_name='data', index_col=0)
    if cuenca in ['Rapel', 'Mataquito','Maule']:
        for col in ['Ano','Mes']:
            del df[col]
        df.index = pd.to_datetime(df.index)
    df_estaciones = pd.read_excel(path, sheet_name='info estacion')
    fig = plt.figure(figsize=(11, 17))
    axes = fig.add_subplot(111)
    modules_FAA.plot_diagrama_cruces(df,1979,2019, ax = axes)
    plt.title('Disponibilidad y calidad de datos diarios\nCaudales Río '+ cuenca)
    plt.show()
    fig = plt.figure(figsize=(22, 17))
    axes = fig.add_subplot(121)
    modules_FAA.plot_dataframe_table(df_estaciones[['Nombre estacion', 'Codigo Estacion']].iloc[0:27,:], ax = axes, path = 'Maipo_Estaciones_Q.png', fields = ['Nombre estacion', 'Codigo Estacion'])
    axes2 = fig.add_subplot(122)
    modules_FAA.plot_dataframe_table(df_estaciones[['Nombre estacion', 'Codigo Estacion']].iloc[27:,:], ax = axes2, path = 'Maipo_Estaciones_Q.png', fields = ['Nombre estacion', 'Codigo Estacion'])
    plt.show()


# =======================================     
def hidrograph(cuenca):
# ----------------------------------------
# Esta función cueantifica la hidrografía
# y los cuerpos de agua de la cuenca
    ruta_hidrograph = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\GIS\Hidrografia\Red_Hidrografica.shp'
    ruta_fuentes = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\SIG\Mapoteca DGA\Mapoteca_DGA\01_Carta_Base\Hidrografía\Fuentes_Poly_Completa_fixed.shp'
    ruta_cuencas = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\SIG\Mapoteca DGA\Mapoteca_DGA\02_DGA\Cuencas\Cuencas_DARH_2015_Cuencas.shp'
    ruta_glaciares = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\SIG\Glaciares\IPG2014\IPG2014 Fixed geometries.shp'
    ruta_humedales = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\SIG\Inventario_humedales_publico.gdb\Humedales_ZC.shp'
# ----------------------------------------


    # hidrografia
    hidrograph = geopandas.read_file(ruta_hidrograph)
    hidrograph = hidrograph.to_crs(epsg = 32719)
    
    # fuentes
    fuentes = geopandas.read_file(ruta_fuentes)
    
    # glaciares
    glaciares = geopandas.read_file(ruta_glaciares)
    
    # cuenca
    cuenca_shp = geopandas.read_file(ruta_cuencas)    
    cuenca_shp = cuenca_shp[cuenca_shp['COD_CUENCA'] == dicc_ruta[cuenca][1]]
    
    # humedales
    humedales = geopandas.read_file(ruta_humedales)

    # clip y recuento de hidrografía
    hidrograph_cuenca = geopandas.clip(hidrograph, cuenca_shp)
    hidrograph_dr = hidrograph_cuenca.drop_duplicates(subset='Nombre', keep="first")
    hidrograph_count = hidrograph_dr.groupby(['Dren_Tipo']).agg(['count'])
    hidrograph_count.index = [unidecode(x) for x in hidrograph_count.index] 
    hidrograph_count.to_csv(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\scripts\outputs\hydrograph\hidrograph_groups_'+cuenca+'.csv')

    # clip de fuentes de agua
    fuentes_cuenca =  geopandas.clip(fuentes, cuenca_shp)
    fuentes_dr = fuentes_cuenca.drop_duplicates(subset='NOMBRE', keep="first")
    fuentes_count = fuentes_dr.groupby(['TIPO']).agg(['count'])
    fuentes_count.index = [unidecode(x) for x in fuentes_count.index] 
    fuentes_count.to_csv(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\scripts\outputs\hydrograph\sources_groups_'+cuenca+'.csv')

    # clip de glaciares
    glaciares_cuenca =  geopandas.clip(glaciares, cuenca_shp)
    glaciares_dr = glaciares_cuenca.drop_duplicates(subset='COD_GLA', keep="first")
    glaciares_count = glaciares_dr.groupby(['CLASIFICA']).agg(['count'])
    glaciares_count.index = [unidecode(x) for x in glaciares_count.index] 
    glaciares_count.to_csv(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\scripts\outputs\hydrograph\glaciares_groups_'+cuenca+'.csv')
    
    # clip de humedales
    humedales_cuenca =  geopandas.clip(humedales, cuenca_shp)
    humedales_dr = humedales_cuenca.drop_duplicates(subset='Id_humedal', keep="first")
    humedales_dr['Identificador'] = ''
    humedales_count = humedales_dr.groupby(['Identificador']).agg(['count'])
    humedales_count.index = [unidecode(x) for x in humedales_count.index] 
    humedales_count.to_csv(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\scripts\outputs\hydrograph\humedales_groups_'+cuenca+'.csv')
       
    # graficar todo
    fig = plt.figure(figsize=(11, 8.5))
    axes = fig.add_subplot(111)
    hidrograph_cuenca.plot(ax = axes, alpha = 0.8)
    fuentes_cuenca.plot(ax = axes, color = 'blue')
    cuenca_shp.plot(ax = axes, alpha = 0.8, linewidth = 2, facecolor="none", edgecolor = 'gray')
    glaciares_cuenca.plot(ax = axes, color = 'cyan', edgecolor=None, linewidth = 0.1)
    humedales_cuenca.plot(ax = axes, color = 'green', edgecolor=None, linewidth = 0.1)
    legend = [Line2D([0], [0], color="steelblue", lw=2),
                Patch(facecolor='blue', edgecolor=None, label='Cuerpos de agua'),
                Patch(facecolor='cyan', edgecolor=None, label='Glaciares'),
                Patch(facecolor='green', edgecolor=None, label='Humedales'),
                Patch(facecolor='gray', edgecolor=None, label='Cuenca del Río '+cuenca)]
    axes.legend(legend, ['Hidrografía','Cuerpos de agua','Glaciares','Humedales','Cuenca del Río '+cuenca], loc = 'upper left')
    ctx.add_basemap(ax = axes, crs= cuenca_shp.crs.to_string(),
                        source = ctx.providers.Esri.WorldTerrain, zoom = 8)
    x, y, arrow_length = 0.95, 0.95, 0.07
    axes.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=20,
                xycoords=axes.transAxes)
    x, y, scale_len = cuenca_shp.bounds['minx'], cuenca_shp.bounds['miny'].min(), 20000 #arrowstyle='-'
    scale_rect = matplotlib.patches.Rectangle((x,y),scale_len,200,linewidth=1,
                                            edgecolor='k',facecolor='k')
    axes.add_patch(scale_rect)
    plt.text(x+scale_len/2, y+5000, s='20 KM', fontsize=10,
                 horizontalalignment='center')
    axes.set_xlabel('Coordenada Este WGS84 19S (m)')
    axes.set_ylabel('Coordenada Norte WGS84 19S (m)')

  
# =======================================     
def runoff(cuenca):
# ----------------------------------------
# Esta función grafica las CVE, CD y CMA
# de cada cuenca
    
    # diccionario de caudales en cada caso
    dicc_caudales = {'Maipo' : ['../Etapa 1 y 2/datos/Maipo_cr2corregido_Q.xlsx','Q_relleno_MLR_Maipo_1980-2020_monthly_NAT.csv'], 
                     'Rapel' : ['../Etapa 1 y 2/datos/RIO RAPEL_Q_mensual.xlsx','Q_relleno_MLR_Rapel_1980-2020_monthly_TUR.csv'], 
             'Mataquito' : ['../Etapa 1 y 2/datos/RIO MATAQUITO_mensual.xlsx','Q_relleno_MLR_Mataquito_1980-2020_monthly_NAT_v2.csv'], 
             'Maule' : [r'../Etapa 1 y 2/datos/RIO MAULE_mensual.xlsx','Q_relleno_MLR_Maule_1980-2020_monthly_NAT.csv'],
             'Nuble' : [r'../Etapa 1 y 2/datos/RIO ÑUBLE_Q_mensual.xlsx','Q_mon_Nuble_NAT_flags.csv']}

    # df con Estaciones
    df_estaciones = pd.read_excel(dicc_caudales[cuenca][0], sheet_name='info estacion')
    
    def runoffAgroClima(cuenca):
        # ===================================================
        # Agroclima
        # ---------------------------------------------------
        # rutina para graficar caudales para Agroclima
        # ---------------------------------------------------
    
        estacion = '06028001-0'
        estacion = '06008005-4'
        caudales = modules_CCC.CDA(dicc_caudales[cuenca][1])
        caudales = pd.DataFrame(caudales[estacion])
        
        caudales_nam = modules_CCC.get_names(caudales,df_estaciones[['Nombre estacion', 'Codigo Estacion']])
        caudales_nam.columns = caudales_nam.columns+' R.N.'
        modules_CCC.CMA(caudales_nam, 10 ,22, 1, 1)
    
        sup_regada = pd.read_excel(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos\AgroClima\Tendencia_sup_regada\tendencia_rapel.xlsx', index_col = 0)
        sup_regada.index = [2005,2006,2007,2008,2009,2010,2013,2014,2015,2016,2017,2018,2019,2020]
        
        ax = plt.gca()
        ax1 = ax.twinx()
        sup_regada.iloc[:,0].plot(color='g')
        
        ax.legend(['Caudal','Tendencia'])
        ax1.legend(['Superficie efectivamente regada','Superficie efectiva'], loc='best', bbox_to_anchor=(0.91, 0.92))
        ax1.set_ylabel('Superficie (há?)')
        ax1.set_ylim(bottom = 200000)

    
    caudales = modules_CCC.CDA(dicc_caudales[cuenca][1])
    df = df_estaciones[df_estaciones['Codigo Estacion'].isin(caudales.columns)]
    geo_df = geopandas.GeoDataFrame(df, geometry = geopandas.points_from_xy(df['UTM Este'], df['UTM Norte']))
    geo_df = geo_df.set_crs(epsg = 32719)
    geo_df = geo_df.to_crs(epsg = 4326 )
    geo_df.to_file('.\outputs\caudales\geoest'+cuenca+'.shp')
    
    # graciar hidrogramas
    caudales_nam = modules_CCC.get_names(caudales,df_estaciones[['Nombre estacion', 'Codigo Estacion']])
    caudales_nam.columns = caudales_nam.columns+' R.N.'
    caudales_nam.columns = [x.replace('Rio','Río') for x in caudales_nam.columns]
    if cuenca == 'Maule':
        estaciones = [x for x in caudales_nam if x not in ['RIO MAULE EN DESAGUE LAGUNA DEL MAULE R.N.','RIO MAULE EN LOS BAÑOS R.N.']]
        caudales_nam = caudales_nam[estaciones]
        
    plt.close("all")
    for i in range(4,len(caudales_nam.columns)+4,4):
        # caudales medios anuales
        modules_CCC.CMA(caudales_nam.iloc[:,i-4:i], 10 ,22, 2, 2)
        
        # # curvas de duración de caudales
        # fig, axes = plt.subplots(2,2,figsize=(10, 22))
        # modules_CCC.CDQ(caudales_nam.iloc[:,i-4:i], 4, fig,  axes)
        
        # # curvas de variación estacional
        fig, axes = plt.subplots(2,2,figsize=(10, 22))
        axes = axes.reshape(-1)
        modules_CCC.CVE_1979_2019_mon(caudales_nam.iloc[:,i-4:i], fig, axes, 4, 1979, 2019)

        # anomalias
        fig, axes = plt.subplots(2,2,figsize=(10, 22))
        axes = axes.reshape(-1)
        modules_CCC.ANOM(caudales_nam.iloc[:,i-4:i], 20, 11, 0.72, 0.02, 110, 'MS', fig, axes)
            