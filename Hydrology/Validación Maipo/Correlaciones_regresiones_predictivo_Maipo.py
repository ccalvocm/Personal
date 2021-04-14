# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Author: CCCM

"""

#%% Preambulo

def limpiar_kernel():    
    try:
        from IPython import get_ipython
        get_ipython().magic('clear')
        get_ipython().magic('reset -f')
    except:
        pass
    
# limpiar_kernel()

import pandas as pd
# import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from itertools import cycle
import geopandas 

#funciones

    
def mejoresCorrelaciones(df, col, Nestaciones):
    ordenados = df.sort_values(by=col, ascending = False)
    return ordenados.index[:Nestaciones]

# Función rut
def digito_verificador(rut):
    reversed_digits = map(int, reversed(str(rut)))
    factors = cycle(range(2, 8))
    s = sum(d * f for d, f in zip(reversed_digits, factors))
    if (-s) % 11 > 9:
        return 'K'
    else:
        return (-s) % 11

def extenderQ(dfDGA, dfCR2bueno):
    
    dfDGA_aux = dfDGA.copy()
    dfDGA_aux = dfDGA_aux.loc[dfDGA_aux.index <= '01-01-2008']
    
    for columna in dfDGA.columns:
#        serie_adicional = dfCR2bueno.loc[dfCR2bueno.index.year <= 2007,columna_CR2]
        missing_DGA = dfDGA_aux[columna][dfDGA_aux[columna].isna()]
        dfDGA.loc[missing_DGA.index,columna] = dfCR2bueno.loc[missing_DGA.index,columna]
    return dfDGA

# ================================
#     Encontrar puntos cercanos 
# ================================
    
def min_dist(point, gpd2, n_multivariables):
    gpd2['Dist'] = gpd2.apply(lambda row:  point.distance(row.geometry),axis=1)
    return gpd2.sort_values(by=['Dist']).loc[gpd2.sort_values(by=['Dist']).index[0:n_multivariables],'Codigo Est']
       
    
#%%    
    
def main():    

#%%    
    ruta_GitHub = r'D:\GitHub'
    ruta_GitHub = r'C:\Users\ccalvo\Documents\GitHub'
    ruta_OD = r'E:\CIREN\OneDrive - ciren.cl'
    # ruta_OD = r'C:\Users\ccalvo\OneDrive - ciren.cl'
    ruta_Q = ruta_OD+r'\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos\RIO MAIPO_Q_mensual_Mapocho_Manzano_RN_flags.csv' 
    ruta_Q = ruta_OD+r'\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos\RIO MAIPO_Q_mensual.xlsx' 

    ruta_shp = ruta_OD+r'\Of hidrica\GIS\estacionesMaipo.shp'
    
    est_geo = geopandas.read_file(ruta_shp)
    Q_monthly = pd.read_excel(ruta_Q, index_col = 0, parse_dates = True)
  

    #meses
    meses = [4,5,6,7,8,9,10,11,12,1,2,3]
    

    year_i = 1979
    year_f = 2020
    
    #fechas
    inicio = pd.to_datetime(str(year_i)+'-03-31',format='%Y-%m-%d')
    fin = pd.to_datetime(str(year_f)+'-03-31',format='%Y-%m-%d')
    Q_monthly = Q_monthly.loc[(Q_monthly.index <= fin) & (Q_monthly.index >= inicio)]  
    
    #Estaciones que no son canales
    estaciones_nocanales = ['05748001-7','05741001-9','05722001-5','05703006-2','05716003-9','05721001-K','05730008-6','05735001-6','05707002-1','05705001-2','05710001-K','05701001-0','05701002-9','05701009-6','05704002-5','05737019-K','05722002-3','05720001-4','05706001-8','05715001-7','05721016-8','05702001-6','05730005-1']
    
    Q_monthly = Q_monthly[[x for x in Q_monthly.columns if x in estaciones_nocanales]]

    #minimo de años con datos
    minYr = 20
    
  #%%Crear indice de fechas, convertir años a int y calcular frecuencia de datos

    data = Q_monthly.notnull().astype('int')
    data = data.groupby(Q_monthly.index.year)  
    data_anual = data.aggregate(np.sum)
    data_anual = data_anual/(12*0.8)  
    data_anual = data_anual.apply(lambda x: [y if y < 1 else 1 for y in x])
    data_anual = data_anual.transpose()
   
    data_anual = data_anual.sort_index()
    estaciones_minimas = pd.DataFrame(data_anual.sum(axis=1), columns = ['registro'])
    estaciones_minimas = estaciones_minimas[estaciones_minimas['registro']>= minYr]
    
    Q_monthly_filtradas = Q_monthly.copy()[estaciones_minimas.index]
    
    #extender pre 1980
    
    Q_cr2_bueno = pd.read_csv(ruta_OD+'\\Of hidrica\\Clima\\Q\\cr2_qflxAmon_2018\\cr2_qflxAmon_2018.txt', sep = ',', na_values=["-9999"], index_col = 0)[[x[1:-2] for x in Q_monthly_filtradas]].iloc[14:]
    Q_cr2_bueno.index = pd.to_datetime(Q_cr2_bueno.index) 
    Q_cr2_bueno = Q_cr2_bueno[((Q_cr2_bueno.index > inicio) & (Q_cr2_bueno.index <= fin))]
    
    for x in Q_cr2_bueno.columns:
        Q_cr2_bueno.rename(columns={x: '0'+str(x)+'-'+str(digito_verificador(x))}, inplace=True)  
    
    Q_monthly_filtradas = extenderQ(Q_monthly_filtradas,Q_cr2_bueno)
    
    
    fig, ax = plt.subplots(7,5)
    ax = ax.reshape(-1)
    for i, col in enumerate(Q_cr2_bueno.columns):
        pd.DataFrame(Q_cr2_bueno[col], dtype = float).plot(ax = ax[i], legend = False, color = 'r', linewidth = 3)
        pd.DataFrame(Q_monthly_filtradas[col], dtype = float).plot(ax = ax[i], legend = False, color = 'b')
        ax[i].set_title(col)
        ax[i].set_ylabel('Q $m^3/s$')

    #%% Multivariable
    
    n_multivariables = 4
    
    stdOutliers = 3.129
    
           
    Q_monthly_MLR = Q_monthly_filtradas.copy()

    estaciones = Q_monthly_filtradas.columns
        
    # actualizacióin BHN
#    estaciones = ['05722002-3','05748001-7']    
    
    for ind,col in enumerate(estaciones):
        
        print('Rellenando estación '+str(col))
                    
        for mes in meses:
            
                 
            Q_monthly_mes = Q_monthly_filtradas.loc[Q_monthly_filtradas.index.month == mes].copy()
                        
            y = Q_monthly_mes[col]
                       
            if y.count() < 1:
            
                continue
            
            correl = Q_monthly_mes.astype(float).corr()
            est_indep = mejoresCorrelaciones(correl, col, n_multivariables)
            est_cercanas = min_dist(est_geo[est_geo['Codigo Est'] == col].geometry,
                        est_geo[est_geo['Codigo Est'].isin(estaciones)],n_multivariables)
            est_indep = list(set(est_indep) & set(est_cercanas))
            x = pd.DataFrame(Q_monthly_mes.loc[Q_monthly_mes.index.month == mes][est_indep].copy(),dtype = float)
#            
            x = x[np.abs(x-x.mean())<=(stdOutliers*x.std())]
            x = x[x[x.count().idxmax()].notna()]                    

            est_na = x.count()[x.count() == 0].index.values.tolist()
            
            x = x.drop(est_na, axis = 1)
            
            max_value_ = x.mean()+stdOutliers*x.std()
            
            imp = IterativeImputer( imputation_order='random', random_state=0, max_iter=25, min_value = 0, max_value = max_value_, sample_posterior = True)
            Y = imp.fit_transform(x)
            Q_monthly_MLR_mes = pd.DataFrame(Y, columns = x.columns, index = x.index )
            Q_monthly_MLR_mes = Q_monthly_MLR_mes.dropna()

            Q_monthly_MLR.loc[Q_monthly_MLR_mes.index,col] = Q_monthly_MLR_mes[col]

            Q_monthly_MLR.loc[Q_monthly_mes.index,col] = Q_monthly_MLR.loc[Q_monthly_mes.index,col].fillna(Q_monthly_MLR.loc[Q_monthly_mes.index,col].median())

#Graficar
    nticks = 1
    plt.close("all")
    fig, ax  = plt.subplots(7,5)
    ax = ax.reshape(-1)
    Q_monthly_filtradas = pd.DataFrame(Q_monthly_filtradas, dtype = float)
    Q_monthly_MLR = pd.DataFrame(Q_monthly_MLR, dtype = float)
    diff = Q_monthly_MLR - Q_monthly_filtradas
    diff.index.names = ['']
    logplot = False

    for ind,col in enumerate(estaciones):
#        fig.add_subplot(2,1,ind+1)
        Q_monthly_MLR_sim = Q_monthly_MLR[col].copy()
        Q_monthly_MLR_sim.index.names = ['']
        Q_monthly_MLR_sim.loc[Q_monthly_filtradas[col][Q_monthly_filtradas[col].isna()].index] = np.nan
        Q_monthly_MLR_sim.plot(ax = ax[ind], linewidth = 3, logy = logplot)
        Q_monthly_filtradas[col].plot(ax = ax[ind], linewidth = 1, logy = logplot)
        diff[col].plot(ax = ax[ind], linewidth = 4, color = 'k', logy = logplot)
        ticks = ax[ind].xaxis.get_ticklocs()[::nticks]
        fig.canvas.draw()
        ticklabels = [l.get_text() for l in ax[ind].xaxis.get_ticklabels()][::nticks]
#        
        ax[ind].xaxis.set_ticks(ticks)
        ax[ind].xaxis.set_ticklabels(ticklabels)
        ax[ind].figure.show()
        plt.ylabel('Q $m^3/s$')
        plt.title('Estación '+col)
        ax[ind].set_ylim(bottom = 0)
    plt.legend(['Predictor','Original','Residual'],bbox_to_anchor=(1.05, 1), loc='upper left')    
    Q_monthly_MLR.to_csv(ruta_OD+'\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos\Q_relleno_MLR_Maipo_'+str(year_i+1)+'-'+str(year_f)+'_monthly_OBS.csv')


