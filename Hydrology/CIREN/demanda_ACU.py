# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 19:23:12 2021

@author: Carlos
"""

# ==============================================
# librerias 
# ----------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas
from statsmodels.tsa.arima.model import ARIMA
import contextily as ctx
from matplotlib.lines import Line2D
import matplotlib
from matplotlib.patches import Patch
# ----------------------------------------------


# ==============================================
def pronostico_ARMA(df,meses, orden):
# ----------------------------------------------
# pronostico ARMA, recibe:
# df : dataframe
# meses : int de meses a pronosticar
# orden: tuple del orden del modelo ARMA    
# ----------------------------------------------

    # fit model
    model = ARIMA(df, order=orden)
    model_fit = model.fit()
    # make prediction
    return model_fit.forecast(meses), model_fit.get_forecast(meses).conf_int(alpha=0.3)


# ==============================================
def ddaActual():
# ----------------------------------------------
# demanda actual
# recibe la ruta de los datos de producción
# ----------------------------------------------

    # ------------rutas
    ruta = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Demanda\ACU\01 DDA ACU ACT_FUT_XIII_VII PRO-CCALVO.xlsx'
    ruta_shp = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Solicitudes de información\Recibidos\SERNAPESCA\Pisciculturas SIAC\Pisciculturas SIAC.shp'
        
    # --------cargar pisciultoras
    geodf = geopandas.read_file(ruta_shp)
    geodf = geodf.rename({'X': 'Lon', 'Y': 'Lat', 'CENTRO' : 'Código Centro',
                          'Nm_Comuna' : 'Comuna', 'Nm_Sector' : 'Sector', 'dbo_Cc_Cue' : 'CuerpoAgua'}, axis=1)
    
    # ----------produccion
    prod = pd.read_excel(ruta, sheet_name = 'Producc Anual + DDA Actual SUBC', skiprows = 2)
    prod = prod[prod['Región '].isin([6,7,13])]
    del prod['Año 2014 orig']
    del prod['Año 2015 orig']
    prod = prod.drop(prod['Centro '][prod['Centro '].isin([70013,130005])].index)
    
    prod['Total general'] = prod.iloc[:,12:34].sum(axis = 1, skipna = True)
    for j in range(2016,2022):
        prod['DDA (MMm3/año '+str(j)] = np.nan
        
    # ----------coordenadas de centros productivos
    coord = pd.read_excel(ruta, sheet_name = 'Coordenadas ', index_col = 6)
    coord = coord[coord['Región'].isin([6,7,13])]
    coord = coord[~coord.index.duplicated(keep='first')]
    coord = coord.loc[prod['Centro '].astype('int32')]
    
    # -----centros que no están en el shp 
    geodf_extra = coord.loc[coord.index.isin(list(set(coord.index)-set(geodf['Código Centro'].values)))]
    geodf_extra = geodf_extra.reset_index()
    geodf_extra.index = geodf_extra.index+13
    geodf_extra['Lon'] = -geodf_extra['Lon']
    geodf_extra['Lat'] = -geodf_extra['Lat']
    geodf_concat = pd.concat([geodf, geodf_extra])
    geodf_concat = geodf_concat[~geodf_concat['Código Centro'].duplicated(keep='first')]
    geodf_concat = geopandas.GeoDataFrame(
    geodf_concat, geometry=geopandas.points_from_xy(geodf_concat.Lon, geodf_concat.Lat))
        
    # ----------tasas de consumo
    tasas = pd.read_excel(ruta, sheet_name = 'Tasas ordenadas', index_col = 0).loc[prod['Especie '],'Valor (m3/ton)']  
    
    # ---------recirculacion
    r_salmon = 18_000
    r_trucha = 10_800
    
    # ----------tasa por produccion, descontando la recirculacion
    for i in range(12,34):
        tasas.loc[tasas.index.str.contains('SALMON')] = tasas.loc[tasas.index.str.contains('SALMON')].values-r_salmon*max(0,i-28)
        tasas.loc[tasas.index.str.contains('TRUCHA')] = tasas.loc[tasas.index.str.contains('TRUCHA')].values-r_trucha*max(0,i-28)
        dda = prod.iloc[:,i].multiply(tasas.values)/1e6
        prod.iloc[:,i+23] = dda.values
        
    # ----------proyeccion a 2021
    prod_h = prod.copy().iloc[:,12+23:34+23].transpose().replace(np.nan,0.)
    
    for idx in prod.itertuples():
        prod.loc[idx[0],'DDA (MMm3/año 2021'] = max(pronostico_ARMA(prod_h[idx[0]][-10:],1,(1,2,4))[0].values,0)

    # -----------QAQC y guardar
    prod.copy().iloc[:,12+23:34+23].transpose().plot()
    prod.to_csv(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Demanda\ACU\demanda_ACU_2020.csv')
    geodf_concat = geodf_concat.set_crs(epsg=4326)
    geodf_concat = geodf_concat.to_crs(epsg=32719)
    geodf_concat.to_file(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Demanda\ACU\demanda_ACU_2020.shp',
                         driver='ESRI Shapefile',encoding = 'utf-8')
    
    fig = plt.figure(figsize=(11, 8.5))
    axes = fig.add_subplot(111)
    geodf_concat.plot(ax = axes)
    cuencas = geopandas.read_file('..//Etapa 1 y 2//GIS//Cuencas_DARH//Cuencas//Cuencas_DARH_2015.shp')
    cuencas = cuencas.loc[(cuencas['COD_CUENCA'] == '0703') | (cuencas['COD_CUENCA'] == '0701') | (cuencas['COD_CUENCA'] == '0600') | (cuencas['COD_CUENCA'] == '1300')]
    cuencas.plot(ax = axes, alpha=0.4)
    ctx.add_basemap(ax = axes, crs= geodf_concat.crs.to_string(),
                            source = ctx.providers.Esri.WorldTerrain, zoom = 8)
    
    x, y, arrow_length = 0.9, 0.95, 0.1
    axes.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20,
            xycoords=axes.transAxes)
    
    # ----------mensual
    cosechas = {2012 : '2012_cosechas_cc_mes_1.xls',
                2013 : '2013_cosechas_cc_mes_1.xls',
                2014 : '2014_cosechas_cc_mes_1.xls',
                2015 : '2015_cosechas_cc_mes_1.xls',
                2016 : '2016_cosechas_cc_mes_1.xls',
                2017 :  '2017_cosechas_cc_mes_1.xls',
                2018 : '2018_cosechas_cc_mes_1.xls'}
    root = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Solicitudes de información\Recibidos\SERNAPESCA'
   
    writer = pd.ExcelWriter(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Demanda\ACU\demanda_ACU_2021_mon.xlsx', engine='xlsxwriter')
                

    fig, ax = plt.subplots(5,5)
    ax = ax.reshape(-1)
    
    for ind,yr_ind in enumerate(range(1999,2022)):
        if yr_ind < 2012:
            yr = 2012
        elif yr_ind > 2018:
            yr = 2018
        else:
            yr = yr_ind
            
        file = cosechas[yr]
        archivo = pd.read_excel(root+'\\'+file, skiprows = 4, index_col = 0)
        archivo = archivo.reset_index().dropna().set_index('ESPECIE')
        prod_aux = prod['Especie '].copy().replace(['BAGRE AGUA DULCE','ESTURION OSETRA','ESTURION DE SIBERIA','ESTURION BLANCO'],'TOTAL PECES')
        prod_aux = prod_aux.replace(prod_aux[prod_aux.str.contains('TRUCHA')].iloc[0], archivo.index[archivo.index.str.contains('TRUCHA')][0])
        prod_aux = prod_aux.replace(prod_aux[prod_aux.str.contains('SALMON PLATEADO')].iloc[0], archivo.index[archivo.index.str.contains('SALMON PLATEADO')][0])
        prod_aux = prod_aux.replace(prod_aux[prod_aux.str.contains('SALMON PLATEADO')].iloc[0], prod_aux[prod_aux.str.contains('SALMON DEL ATLANTICO')].iloc[0])
        prod_mon = archivo.loc[prod_aux].replace('-',0).astype(float)
        prod_mon['Total'] = prod_mon.sum(axis = 1)
        prod_mon = prod_mon.iloc[:,0:-1].div(prod_mon.iloc[:,0:-1].sum(axis = 1), axis=0)
            
        # ------Write each dataframe to a different worksheet.
        
        diasmes = pd.date_range(str(yr_ind)+'-01-01',str(yr_ind)+'-12-01',freq='MS').daysinmonth
        prod_ponderada = prod_mon.multiply(prod.iloc[:,ind+12+23].values, axis="index")
        prod_ponderada = prod_ponderada.div(diasmes)*1e6/86400
        prod_ponderada.index = prod.index
        prod_ponderada.transpose().loc[['ABR', 'MAY', 'JUN', 'JUL', 'AGO','SEP', 'OCT', 'NOV', 'DIC','ENE', 'FEB', 'MAR']].plot(ax = ax[ind], legend = False)
        ax[ind].set_title(str(yr_ind))
        ax[ind].set_ylabel('Q $m^3/s$')

        prod_mon = prod.copy().iloc[:,0:12]
        prod_mon[prod_ponderada.columns] = prod_ponderada
        
        prod_mon.to_excel(writer, sheet_name=str(yr_ind))
        
    for i in [23,24]:
        fig.delaxes(ax[i])
    # ------ Close the Pandas Excel writer and output the Excel file.
    writer.save()
    writer.close()

def postproceso():
    # rutas
    path = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Demanda\ACU\demanda_ACU_2021_mon.xlsx'
    
    plt.close("all")
    # plot centros acuícolas
    fig = plt.figure(figsize=(11, 8.5))
    axes = fig.add_subplot(111)
    pc = geopandas.read_file('..//SIG//Pisciculturas SIAC//Centros_acuícolas_2021.shp')
    
    cuencas = geopandas.read_file('..//Etapa 1 y 2//GIS//Cuencas_DARH//Cuencas//Cuencas_DARH_2015.shp')
    cuencas = cuencas.loc[(cuencas['COD_CUENCA'] == '0703') | (cuencas['COD_CUENCA'] == '0701') | (cuencas['COD_CUENCA'] == '0600') | (cuencas['COD_CUENCA'] == '1300')]
    cuencas.plot(ax = axes, alpha=0.4)
    pc.plot(ax = axes, color = 'blue')
    ctx.add_basemap(ax = axes, crs= pc.crs.to_string(),
                        source = ctx.providers.Esri.WorldTerrain, zoom = 8)
    axes.legend(['Centros acuícolas en tierra'], loc = 'upper left')
    x, y, arrow_length = 0.95, 0.95, 0.07
    axes.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=20,
                xycoords=axes.transAxes)
    x, y, scale_len = cuencas.bounds['minx'], cuencas.bounds['miny'].min(), 20000 #arrowstyle='-'
    scale_rect = matplotlib.patches.Rectangle((x.iloc[0],y),scale_len,200,linewidth=1,
                                            edgecolor='k',facecolor='k')
    axes.add_patch(scale_rect)
    plt.text(x.iloc[0]+scale_len/2, y+5000, s='20 KM', fontsize=10,
                 horizontalalignment='center')
    axes.set_xlabel('Coordenada Este WGS84 19S (m)')
    axes.set_ylabel('Coordenada Norte WGS84 19S (m)')

    plt.close("all")
    # calcular la demanda por cuenca
    years = range(1999,2022)
    dda_cuenca = pd.DataFrame([], columns = [57,60,73], index =  pd.date_range('1999-01-01','2021-12-01',freq = 'MS') )
    for yr in years:
        dda_ACU_yr = pd.read_excel(path, sheet_name = str(yr))
        a = dda_ACU_yr.groupby(['COD_CUEN']).sum()
        a = a[['ENE', 'FEB', 'MAR',
       'ABR', 'MAY', 'JUN', 'JUL', 'AGO', 'SEP', 'OCT', 'NOV', 'DIC']]
        
        for cuenca in dda_cuenca.columns:
            dda_cuenca.loc[dda_cuenca.index.year == yr,cuenca] = a.loc[cuenca].values
    
    cuencas = ['Río Maipo','Río Rapel','Río Maule']
    fig, axes = plt.subplots(2,2,figsize=(10, 22))
    axes = axes.reshape(-1)
    for i in range(0,3):
        # plt.figure()
        dda_cuenca.iloc[:,i].plot(ax = axes[i])
        axes[i].set_title('Demanda de centros acuícolas en la cuenca del '+cuencas[i])
        axes[i].set_ylabel('Demanda media mensual ($m^3/s$)')
        axes[i].grid()

if __name__ == '__main__':
    ddaActual()
        

