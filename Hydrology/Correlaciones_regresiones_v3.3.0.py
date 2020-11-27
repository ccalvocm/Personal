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
import gc

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
    ruta_GitHub = r'D:\GitHub'
    # ruta_GitHub = r'C:\Users\ccalvo\Documents\GitHub'

#    ruta_Q = ruta_GitHub+r'\Analisis-Oferta-Hidrica\DGA\datosDGA\Q\Maule\Q_Maule_1900-2020_v0.csv'
    # ruta_Q = ruta_GitHub+r'\Analisis-Oferta-Hidrica\DGA\datosDGA\Q\Maipo\RIO MAIPO_Q_diario.csv'
    # ruta_Q = ruta_GitHub+r'\Analisis-Oferta-Hidrica\Hidrología\Caudales\Validacion\cr2_Maipo_Q.xlsx'
    ruta_Q = r'C:\Users\Carlos\Documents\Python Scripts\cr2_Maipo_Q.xlsx'
    Q_daily = pd.read_excel(ruta_Q, index_col = 0)
    # Q_daily = pd.read_csv(ruta_Q, index_col = 0)
    Q_daily.index = pd.to_datetime(Q_daily.index)
    
    # ver entrada gráfica
#    Q_daily.plot()
    
    #meses
    meses = [4,5,6,7,8,9,10,11,12,1,2,3]
    
    #fechas
    # inicio = pd.to_datetime('2000-12-31',format='%Y-%m-%d')
    inicio = pd.to_datetime('1949-12-31',format='%Y-%m-%d')
    fin = pd.to_datetime('2007-12-31',format='%Y-%m-%d')
    # fin = pd.to_datetime('2020-01-01',format='%Y-%m-%d')
    Q_daily = pd.DataFrame(Q_daily[Q_daily.index <= fin ],  index = pd.date_range(inicio, fin, freq='D', closed='right'))

    #minimo de años con datos
    minYr = 1

  #%%Crear indice de fechas
    
    #Convertir años a int y calcular frecuencia de datos para mapa de colores

    data = Q_daily.notnull().astype('int')
    data = data.groupby(Q_daily.index.year)  
    data_anual = data.aggregate(np.sum)
    data_anual = data_anual/(365*0.8)  
    data_anual = data_anual.apply(lambda x: [y if y < 1 else 1 for y in x])
    data_anual = data_anual.transpose()
   
    data_anual = data_anual.sort_index()
    estaciones_minimas = pd.DataFrame(data_anual.sum(axis=1), columns = ['registro'])
    estaciones_minimas = estaciones_minimas[estaciones_minimas['registro']>= minYr]
    
    # estaciones_minimas = ['05710001-K', '05701001-0', '05701002-9', '05702001-6', '05704002-5', '05705001-2', '05706001-8', '05707002-1', '05721001-K',
                         # '05722001-5',  '05722002-3', '05716001-2', '05735001-6', '05737002-5','05741001-9', '05746001-6',  '05748001-7']
                   
    # Q_daily_filtradas = Q_daily[estaciones_minimas]

    Q_daily_filtradas = Q_daily[estaciones_minimas.index]
    
    
     #%%Calcular estadígrafos
  
  
    Q_month_mean = Q_daily_filtradas.groupby(Q_daily.index.month).mean()    
    Q_month_std = Q_daily_filtradas.groupby(Q_daily.index.month).std()    
    Q_month_mean = (Q_month_mean.fillna(method='ffill') + Q_month_mean.fillna(method='bfill'))/2
    Q_month_std = (Q_month_std.fillna(method='ffill') + Q_month_std.fillna(method='bfill'))/2
    
    #%% Relleno con OLR
    
    coef_m_mensuales = pd.DataFrame( index = meses, columns = Q_daily_filtradas.columns)
    coef_n_mensuales = pd.DataFrame( index = meses, columns = Q_daily_filtradas.columns)
    coef_r2_mensuales = pd.DataFrame( index = meses, columns = Q_daily_filtradas.columns)
       
    # Est con mejor correlación diaria
    correl = Q_daily_filtradas.corr()
    correl = correl.replace(1,-9999)
    idx = correl.idxmax()
#    idx_2 = correl[col].nlargest(3)
#    r = correl.max()
    Q_daily_mon = Q_daily_filtradas.groupby(Q_daily_filtradas.index.month)
    
    for indice in idx.index:
        print(indice)
        for mes in meses:
            y = Q_daily_mon[indice].apply(list).loc[mes] #mes 1
            est_indep = idx.loc[indice]
            x =  Q_daily_mon[est_indep].apply(list).loc[mes]  #mes 1
            try:
                M, N, R2 = regresion(x,y)
                coef_m_mensuales.loc[mes][indice] = M
                coef_n_mensuales.loc[mes][indice] = N
                coef_r2_mensuales.loc[mes][indice] = R2
            except:
                print('No hay datos para el mes '+str(mes))
        
    Q_daily_rellenas = Q_daily_filtradas.copy()
    coeficientes = pd.DataFrame(index=Q_daily_rellenas.index, columns = ['m','n'])
        
    nticks = 4
    plt.close("all")
    fig = plt.figure()
    for ind,col in enumerate(Q_daily_rellenas.columns):
        print(col)
        missingData = Q_daily_filtradas[col].isna()
        coeficientes['m'] = coef_m_mensuales.loc[Q_daily_filtradas.index.month][col].to_list()
        coeficientes['n'] = coef_n_mensuales.loc[Q_daily_filtradas.index.month][col].to_list()
        Q_x =  Q_daily_filtradas[idx.loc[col]]
        Q_daily_rellenas.loc[missingData,col] = Q_x.loc[missingData]*coeficientes['m'].loc[missingData]+ coeficientes['n'].loc[missingData]
        Q_daily_rellenas.loc[:,col][Q_daily_rellenas.loc[:,col] < 0] = 0
        fig.add_subplot(9,6,ind+1)
        ax1 = Q_daily_rellenas[col].plot()
        Q_daily_filtradas[col].plot(ax = ax1)
        
        ticks = ax1.xaxis.get_ticklocs()[::nticks]
        fig.canvas.draw()
        ticklabels = [l.get_text() for l in ax1.xaxis.get_ticklabels()][::nticks]
        
        ax1.xaxis.set_ticks(ticks)
        ax1.xaxis.set_ticklabels(ticklabels)
        ax1.figure.show()
    plt.legend(['Rellenas','Originales'],bbox_to_anchor=(1.05, 1), loc='upper left')
          
    #%% Multivariable
    
    n_multivariables = 25
    
    stdOutliers = 3.
    
    learnt_month = {x : '' for x in meses}
       
    Q_daily_MLR = Q_daily_filtradas.copy()
        
    estaciones = ['05710001-K', '05722001-5', '05722002-3', '05737002-5', '05716001-2','05748001-7', '05746001-6']

    for ind,col in enumerate(estaciones):
        
        print(col)
        
#        if col in ['05722001-5']:
#            stdOutliers = 3.
#            
#            Q_daily_MLR.loc[Q_daily_MLR.index, col] = Q_daily_MLR[np.abs(Q_daily_MLR[col]-Q_daily_MLR[col].mean())<=(stdOutliers*Q_daily_MLR[col].std())][col]
        
#        if col in ['05746001-6']:
#            stdOutliers = 3.
            
#            Q_daily_MLR.loc[Q_daily_MLR.index, col] = Q_daily_MLR[np.abs(Q_daily_MLR[col]-Q_daily_MLR[col].mean())<=(stdOutliers*Q_daily_MLR[col].std())][col]
            
#        else:
#            stdOutliers = 3.
            
#            Q_daily_MLR.loc[Q_daily_MLR.index, col] = Q_daily_MLR[np.abs(Q_daily_MLR[col]-Q_daily_MLR[col].mean())<=(stdOutliers*Q_daily_MLR[col].std())][col]

        for mes in meses:
            
            if col in learnt_month[mes]:
                
                continue
            
            else:
            
#                print(str(mes))
                 
                Q_daily_mes = Q_daily_filtradas.loc[Q_daily_filtradas.index.month == mes].copy()
                
                y = Q_daily_mes[col]
                           
                if y.count() < 1:
                
                    continue
                
                correl = Q_daily_mes.corr()
                est_indep = mejoresCorrelaciones(correl, col, n_multivariables)
                x = Q_daily_mes.loc[Q_daily_mes.index.month == mes][est_indep.to_list()].copy()
                x = x[np.abs(x-x.mean())<=(stdOutliers*x.std())]
                
                try:
                    x.loc[x.index,'05746001-6'] = x[np.abs(x-x.mean())<=(.8*x.std())]['05746001-6']
                except:
                    err = 'No hay información en el mes'

                x = x[x[x.count().idxmax()].notna()]                 
                
                est_na = x.count()[x.count() == 0].index.values.tolist()
                
                x = x.drop(est_na, axis = 1)
                
                max_value_ = x.mean()+stdOutliers*x.std()
                
#                max_value_.loc[est_na] = Q_month_mean[est_na].loc[mes]+stdOutliers*Q_month_std[est_na].loc[mes]                

#                imp = IterativeImputer(max_iter=1, random_state=0, min_value = 0, sample_posterior = True, verbose = 2)
                imp = IterativeImputer(max_iter=10, random_state=0, min_value = 0, max_value = max_value_, sample_posterior = True)
    #             imp = IterativeImputer(max_iter=1, random_state=0, min_value = 0, max_value = y.mean()+stdOutliers*y.std(), sample_posterior = True, skip_complete = True)
                # Q_daily_MLR_mes = x[x[x.count().idxmax()].notna()]
                
    #            imp.fit(x.values)
#                Y = imp.fit_transform(x.values)
                Y = imp.fit_transform(x)
                Q_daily_MLR_mes = pd.DataFrame(Y, columns = x.columns, index = x.index )
                Q_daily_MLR_mes = Q_daily_MLR_mes.dropna()
                # Y = pd.DataFrame(Q_daily_MLR_mes[col])
                
                Q_daily_MLR.loc[Q_daily_MLR_mes.index,Q_daily_MLR_mes.columns] = Q_daily_MLR_mes[Q_daily_MLR_mes.columns]
                # Q_daily_MLR.loc[Q_daily_MLR_mes.index,Q_daily_MLR_mes.columns] = Q_daily_MLR_mes[Q_daily_MLR_mes.columns]
    #            Q_daily_MLR.loc[Q_daily_mes.index,col] = Q_daily_MLR.loc[Q_daily_mes.index,col].fillna(Q_daily_MLR.loc[Q_daily_mes.index,col].median())
                # Q_daily_MLR.loc[Q_daily_mes.index,col] = Q_daily_MLR.loc[Q_daily_mes.index,col].fillna(Q_daily_MLR.loc[Q_daily_mes.index,col].rolling(60).mean())
#                learnt_month[mes] = Q_daily_MLR_mes.columns.to_list()

#                learnt_month[mes] = Q_daily_MLR_mes.columns.to_list()

                Q_daily_MLR.loc[Q_daily_mes.index,col] = Q_daily_MLR.loc[Q_daily_mes.index,col].fillna(Q_daily_MLR.loc[Q_daily_mes.index,col].median())


            # gc.collect()
            # del imp
            # del A
        
#Graficar
    nticks = 2
    plt.close("all")
    fig = plt.figure()
    for ind,col in enumerate(estaciones):
        fig.add_subplot(8,4,ind+1)
        # mask = Q_daily_filtradas[col].isna()
        # aux = Q_daily_MLR[col]
        # aux[mask] = np.nan
        # ax1 = aux.plot(linewidth = 3)
        ax1 = Q_daily_MLR[col].plot(linewidth = 3)
        Q_daily_filtradas[col].plot(ax = ax1, linewidth = 1)
        # diff = Q_daily_filtradas[col] - aux
        # diff.plot(linewidth = 3)
        
        # plt.ylim([-1,1])
        ticks = ax1.xaxis.get_ticklocs()[::nticks]
        fig.canvas.draw()
        ticklabels = [l.get_text() for l in ax1.xaxis.get_ticklabels()][::nticks]
        
        ax1.xaxis.set_ticks(ticks)
        ax1.xaxis.set_ticklabels(ticklabels)
        ax1.figure.show()
        plt.ylabel('$\Delta$ Q $m^3/s$')
        plt.title('Estación '+col)
    plt.legend(['Predictor','Original'],bbox_to_anchor=(1.05, 1), loc='upper left')    
    Q_daily_MLR.to_csv('Q_relleno_MLR_Maipo_1950-2008_outlier_in_correction.csv')

