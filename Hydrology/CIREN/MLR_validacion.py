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
    
limpiar_kernel()

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#funciones

def regresion(x_,y_):
    X = sm.add_constant(x_)
    resultados_fit = sm.OLS(y_,X,missing='drop').fit()
    N = resultados_fit.params[0]
    M = resultados_fit.params[1]
    R2 = resultados_fit.rsquared
    return [M,N,R2]
    
def mejoresCorrelaciones(df, col, Nestaciones):
    ordenados = df.sort_values(by=col, ascending = False)
    return ordenados.index[:Nestaciones]
    
#%%    
    
def main():    

#%%    
#    ruta_GitHub = r'D:\GitHub'
    ruta_GitHub = r'C:\Users\ccalvo\Documents\GitHub'

#    ruta_Q = ruta_GitHub+r'\Analisis-Oferta-Hidrica\DGA\datosDGA\Q\Maule\Q_Maule_1900-2020_v0.csv'
    ruta_Q = ruta_GitHub+r'\Analisis-Oferta-Hidrica\DGA\datosDGA\Q\Maipo\Maipo_cr2corregido_Q.xlsx'
#    ruta_Q = ruta_GitHub+r'C:\Users\ccalvo\Documents\GitHub\Analisis-Oferta-Hidrica\DGA\datosDGA\Q\Rapel\Rapel_cr2corregido_Q.xlsx'
    Q_daily = pd.read_excel(ruta_Q, index_col = 0)
    Q_daily.index = pd.to_datetime(Q_daily.index)
    

    #meses
    meses = [4,5,6,7,8,9,10,11,12,1,2,3]
    
    year_i = 1949
    year_f = 2001
    
    #fechas
    inicio = pd.to_datetime(str(year_i)+'-12-31',format='%Y-%m-%d')
    fin = pd.to_datetime(str(year_f)+'-12-31',format='%Y-%m-%d')
    Q_daily = pd.DataFrame(Q_daily[Q_daily.index <= fin ],  index = pd.date_range(inicio, fin, freq='D', closed='right'))

    #minimo de años con datos
    minYr = 5

  #%%Crear indice de fechas, convertir años a int y calcular frecuencia de datos

    data = Q_daily.notnull().astype('int')
    data = data.groupby(Q_daily.index.year)  
    data_anual = data.aggregate(np.sum)
    data_anual = data_anual/(365*0.8)  
    data_anual = data_anual.apply(lambda x: [y if y < 1 else 1 for y in x])
    data_anual = data_anual.transpose()
   
    data_anual = data_anual.sort_index()
    estaciones_minimas = pd.DataFrame(data_anual.sum(axis=1), columns = ['registro'])
    estaciones_minimas = estaciones_minimas[estaciones_minimas['registro']>= minYr]
    
    Q_daily_filtradas = Q_daily[estaciones_minimas.index]

    #%% Multivariable
    
    n_multivariables = len(Q_daily_filtradas.columns)
    
    stdOutliers = 3.
    
           
    Q_daily_MLR = Q_daily_filtradas.copy()
#    
    estaciones = ['05710001-K','05701001-0', '05701002-9','05702001-6',
       '05704002-5',
       '05705001-2',
       '05706001-8',
       '05707002-1',
       '05721001-K',
       '05722001-5',
       '05722002-3',
       '05716001-2',
       '05735001-6',
       '05737002-5',
       '05741001-9',
       '05746001-6',
       '05748001-7']

    # actualizacióin BHN
    
    for ind,col in enumerate(estaciones):
        
        print('Rellenando estación '+str(col))
        
        stdOutliers = 3.
            
        for mes in meses:
            
                 
            Q_daily_mes = Q_daily_filtradas.loc[Q_daily_filtradas.index.month == mes].copy()
                        
            y = Q_daily_mes[col]
                       
            if y.count() < 1:
            
                continue
            
            correl = Q_daily_mes.corr()
            est_indep = mejoresCorrelaciones(correl, col, n_multivariables)
            x = Q_daily_mes.loc[Q_daily_mes.index.month == mes][est_indep.to_list()].copy()
#            

                
            x = x[np.abs(x-x.mean())<=(stdOutliers*x.std())]
            x = x[x[x.count().idxmax()].notna()]                 

            est_na = x.count()[x.count() == 0].index.values.tolist()
            
            x = x.drop(est_na, axis = 1)
            
            max_value_ = x.mean()+stdOutliers*x.std()
            
            imp = IterativeImputer( imputation_order='random', random_state=0, max_iter=15, min_value = 0, max_value = max_value_, sample_posterior = True)
            Y = imp.fit_transform(x)
            Q_daily_MLR_mes = pd.DataFrame(Y, columns = x.columns, index = x.index )
            Q_daily_MLR_mes = Q_daily_MLR_mes.dropna()

            Q_daily_MLR.loc[Q_daily_MLR_mes.index,col] = Q_daily_MLR_mes[col]

            Q_daily_MLR.loc[Q_daily_mes.index,col] = Q_daily_MLR.loc[Q_daily_mes.index,col].fillna(Q_daily_MLR.loc[Q_daily_mes.index,col].median())

#Graficar
    nticks = 1
    plt.close("all")
    fig = plt.figure()
    diff = Q_daily_MLR-Q_daily_filtradas
    logplot = False

    for ind,col in enumerate(estaciones):
        fig.add_subplot(6,3,ind+1)
        Q_daily_MLR_sim = Q_daily_MLR[col].copy()
        Q_daily_MLR_sim.loc[Q_daily_filtradas[col][Q_daily_filtradas[col].isna()].index] = np.nan
        ax1 = Q_daily_MLR_sim.plot(linewidth = 3, logy = logplot)
        Q_daily_filtradas[col].plot(ax = ax1, linewidth = 1, logy = logplot)
        diff[col].plot(ax = ax1, linewidth = 4, color = 'k', logy = logplot)
        ticks = ax1.xaxis.get_ticklocs()[::nticks]
        fig.canvas.draw()
        ticklabels = [l.get_text() for l in ax1.xaxis.get_ticklabels()][::nticks]
#        
        ax1.xaxis.set_ticks(ticks)
        ax1.xaxis.set_ticklabels(ticklabels)
        ax1.figure.show()
        plt.ylabel('Q $m^3/s$')
        plt.title('Estación '+col)
        ax1.set_ylim(bottom = 0)
    plt.legend(['Predictor','Original','Residual'],bbox_to_anchor=(1.05, 1), loc='upper left')    
    Q_daily_MLR.to_csv('Q_relleno_MLR_Maipo_'+str(year_i+1)+'-'+str(year_f)+'_outlier_in_correction_median.csv')

