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
#import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
import fiscalyear
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy import stats


#Variables globales
fiscalyear.START_MONTH = 4

#funciones

def agnohidrologico(year_,month_):
    cur_dt = fiscalyear.FiscalDate(year_, month_, 1) 
    retornar = cur_dt.fiscal_year - 1
    return retornar

def regresion(x_,y_):
    X = sm.add_constant(x_)
    resultados_fit = sm.OLS(y_,X,missing='drop').fit()
    N = resultados_fit.params[0]
    M = resultados_fit.params[1]
    R2 = resultados_fit.rsquared
    return [M,N,R2]
    
def MLR(x__,y__): #"multivariable"
    lm = linear_model.LinearRegression()
    model = lm.fit(x__,y__)
    return lm

def LASSO(x__,y__): #"multivariable"
#    alpha = 0.1
    alpha = 0.0001
    lin = Lasso(alpha=alpha,precompute=True,max_iter=1000,
            positive=True, random_state=9999, selection='random')
    lin.fit(x__,y__)
    return lin

def mejoresCorrelaciones(df, col, Nestaciones):
    ordenados = df.sort_values(by=col, ascending = False)
    return ordenados.index[:Nestaciones]
    
#%%    
    
def main():    

#%%    
#    ruta_GitHub = r'D:\GitHub'
    ruta_GitHub = r'C:\Users\ccalvo\Documents\GitHub'

#    ruta_Q = ruta_GitHub+r'\Analisis-Oferta-Hidrica\DGA\datosDGA\Q\Maule\Q_Maule_1900-2020_v0.csv'
    ruta_Q = ruta_GitHub+r'\Analisis-Oferta-Hidrica\DGA\datosDGA\Q\Maipo\RIO MAIPO_Q_diario.csv'
    Q_daily = pd.read_csv(ruta_Q, index_col = 0)
    Q_daily.index = pd.to_datetime(Q_daily.index)
    
    # ver entrada gráfica
#    Q_daily.plot()
    
    #meses
    meses = [4,5,6,7,8,9,10,11,12,1,2,3]
    
    #fechas
#    inicio = pd.to_datetime('2000-12-31',format='%Y-%m-%d')
    inicio = pd.to_datetime('1978-12-31',format='%Y-%m-%d')
    fin = pd.to_datetime('2019-01-01',format='%Y-%m-%d')
    Q_daily = pd.DataFrame(Q_daily[Q_daily.index <= fin ],  index = pd.date_range(inicio, fin, freq='D', closed='right'))

    #minimo de años con datos
    minYr = 15

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
    
    Q_daily_filtradas = Q_daily[estaciones_minimas.index]
    
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
    
    Q_daily_MLR = Q_daily_filtradas.copy()
    
    n_multivariables = 14
    stdOutliers = 3
    
    for ind,col in enumerate(Q_daily_filtradas.columns):
        
        print(col)
        for mes in meses:
            
            Q_daily_mes = Q_daily_filtradas.loc[Q_daily_filtradas.index.month == mes].copy()
            
            y = Q_daily_mes[col].copy()
            correl = Q_daily_mes.corr()
            correl = correl.replace(1,-1e10)
            est_indep = mejoresCorrelaciones(correl, col, n_multivariables)
            x = Q_daily_mes.loc[Q_daily_mes.index.month == mes][est_indep.to_list()]
            x[col] = y
            
            imp = IterativeImputer(max_iter=2, random_state=0, min_value = 0, max_value = y.mean()+stdOutliers*y.std(), sample_posterior = True)
            Q_daily_MLR_mes = x[x[x.count().idxmax()].notna()]
            IterativeImputer(random_state=0)
            imp.fit(Q_daily_MLR_mes.values.T)
            A = imp.transform(Q_daily_MLR_mes.values.T.tolist()).T
            Q_daily_MLR_mes = pd.DataFrame(A, columns = Q_daily_MLR_mes.columns, index = Q_daily_MLR_mes.index )
            Q_daily_MLR_mes = Q_daily_MLR_mes.dropna()
            Y = pd.DataFrame(Q_daily_MLR_mes[col])
            
            
            Q_daily_MLR.loc[Y.index,col] = Y[col]
#            Q_daily_MLR.loc[Q_daily_mes.index,col] = Q_daily_MLR.loc[Q_daily_mes.index,col].fillna(Q_daily_MLR.loc[Q_daily_mes.index,col].median())
            Q_daily_MLR.loc[Q_daily_mes.index,col] = Q_daily_MLR.loc[Q_daily_mes.index,col].fillna(Q_daily_MLR.loc[Q_daily_mes.index,col].rolling(7).mean())

#Graficar
    nticks = 2
    plt.close("all")
    fig = plt.figure()
    for ind,col in enumerate(Q_daily_filtradas.columns):
        fig.add_subplot(8,4,ind+1)
        ax1 = Q_daily_MLR[col].plot(linewidth = 3)
        Q_daily_filtradas[col].plot(ax = ax1, linewidth = 1)
        
        ticks = ax1.xaxis.get_ticklocs()[::nticks]
        fig.canvas.draw()
        ticklabels = [l.get_text() for l in ax1.xaxis.get_ticklabels()][::nticks]
        
        ax1.xaxis.set_ticks(ticks)
        ax1.xaxis.set_ticklabels(ticklabels)
        ax1.figure.show()
        plt.ylabel('Q $m^3/s$')
        plt.title('Estación '+col)
    plt.legend(['Predictor','Original'],bbox_to_anchor=(1.05, 1), loc='upper left')    

