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
# import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from itertools import cycle

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
       
    
#%%    
    
def main():    

#%%    
    ruta_GitHub = r'D:\GitHub'
    # ruta_GitHub = r'C:\Users\ccalvo\Documents\GitHub'
    ruta_OD = r'E:\CIREN\OneDrive - ciren.cl'
    # ruta_OD = r'C:\Users\ccalvo\OneDrive - ciren.cl'
    

    ruta_Q = ruta_GitHub+r'\Analisis-Oferta-Hidrica\DGA\datosDGA\Q\Maule\Q_Maule_1900-2020_v0.csv'

    Q_daily = pd.read_csv(ruta_Q, index_col = 0)
    Q_daily.index = pd.to_datetime(Q_daily.index)
    

    #meses
    meses = [4,5,6,7,8,9,10,11,12,1,2,3]
    

    year_i = 1978
    year_f = 2020
    
    #fechas
    inicio = pd.to_datetime(str(year_i)+'-03-31',format='%Y-%m-%d')
    fin = pd.to_datetime(str(year_f)+'-03-31',format='%Y-%m-%d')
    Q_daily = pd.DataFrame(Q_daily[Q_daily.index <= fin ],  index = pd.date_range(inicio, fin, freq='D', closed='right'))
    
    #Estaciones que no son canales
    
    estaciones_nocanales = ['07340001-5','07335002-6','07308001-0','07381001-9','07375000-8','07378013-6','07378012-8','07341003-7','07354002-K','07354001-1','07356001-2','07355003-3','07355001-7','07355002-5','07350002-8','07331001-6','07339001-K','07336001-3','07306001-K','07372001-K','07379002-6','07320003-2','07379001-8','07307000-7','07379005-0','07374001-0','07357002-6','07359001-9','07350003-6','07350001-K','07351001-5','07308002-9','07321002-K','07322004-1','07300001-7','07383001-K','07322001-7','07303000-5','07317005-2','07317001-K','07317003-6','07332001-1','07335001-8','07330001-0','07335003-4','07341001-0','07343001-1','07358001-3']
    
    Q_daily = Q_daily[[x for x in Q_daily.columns if x in estaciones_nocanales]]

    #minimo de años con datos
    minYr = 20

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
    
    #extender pre 1980
    
    Q_cr2_bueno = pd.read_csv(ruta_OD+'\\Of hidrica\\Clima\\Q\\cr2_qflxDaily_2018\\cr2_qflxDaily_2018.txt', sep = ',', na_values=["-9999"], index_col = 0)[[x[:-2] for x in Q_daily_filtradas]].iloc[14:]
    Q_cr2_bueno.index = pd.to_datetime(Q_cr2_bueno.index) 
    Q_cr2_bueno = Q_cr2_bueno[((Q_cr2_bueno.index > inicio) & (Q_cr2_bueno.index <= fin))]
    
    for x in Q_cr2_bueno.columns:
        Q_cr2_bueno.rename(columns={x: str(x)+'-'+str(digito_verificador(x))}, inplace=True)  
    
    Q_daily_filtradas = extenderQ(Q_daily_filtradas,Q_cr2_bueno)
    
    fig, ax = plt.subplots(5,5)
    ax = ax.reshape(-1)
    for i, col in enumerate(Q_cr2_bueno.columns):
        Q_cr2_bueno[col].plot(ax = ax[i], legend = False, color = 'r', linewidth = 3)
        Q_daily_filtradas[col].plot(ax = ax[i], legend = False, color = 'b')

    #%% Multivariable
    
    n_multivariables = 10
    
    stdOutliers = 3.
    
           
    Q_daily_MLR = Q_daily_filtradas.copy()
#    

# estaciones = ['06003001-4', '06006001-0' , '06011001-8' , '06013001-9', '06018001-6', '06019003-8',
#               '06028001-0', '06027001-5', '06033001-8',  '06034022-6', '06034001-3', '06035001-9', ]


    # estaciones = ['06006001-0', '06003001-4', '06018001-6', '06043001-2', '06011001-8', '06013001-9', '06028001-0', '06027001-5',
    #               '06033001-8', '06034001-3']
    
    # estaciones = ['06003001-4', '06027001-5']
    estaciones = Q_daily_filtradas.columns
        
    Q_daily_MLR = Q_daily_MLR[estaciones]

    # actualizacióin BHN
#    estaciones = ['05722002-3','05748001-7']    
    
    for ind,col in enumerate(estaciones):
        
        print('Rellenando estación '+str(col))
        
        stdOutliers = 30.
            
#        if col in ['05716001-2']:
#            stdOutliers = 1.
#        elif col in ['05748001-7']:
#            stdOutliers = .325
#        elif col in ['05741001-9', '05701001-0']:
#            stdOutliers = np.infty

        for mes in meses:
            
                 
            Q_daily_mes = Q_daily_filtradas.loc[Q_daily_filtradas.index.month == mes].copy()
                        
            y = Q_daily_mes[col]
                       
            if y.count() < 1:
            
                continue
            
            correl = Q_daily_mes.astype(float).corr()
            est_indep = mejoresCorrelaciones(correl, col, n_multivariables)
            x = Q_daily_mes.loc[Q_daily_mes.index.month == mes][est_indep.to_list()].copy()
#            
            x = Q_daily_mes.loc[Q_daily_mes.index.month == mes].copy()

                
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
    diff.index.names = ['']
    logplot = False
    
    fig, ax1 = plt.subplots(5,5)
    ax1 = ax1.reshape(-1)
    for ind,col in enumerate(estaciones):
        Q_daily_MLR_sim = Q_daily_MLR[col].copy()
        Q_daily_MLR_sim.index.names = ['']
        Q_daily_MLR_sim.loc[Q_daily_filtradas[col][Q_daily_filtradas[col].isna()].index] = np.nan
        Q_daily_MLR_sim.plot(ax = ax1[ind], linewidth = 3, logy = logplot)
        Q_daily_filtradas[col].plot(ax = ax1[ind], linewidth = 1, logy = logplot)
        diff[col].plot(ax = ax1[ind], linewidth = 4, color = 'k', logy = logplot)
        ticks = ax1[ind].xaxis.get_ticklocs()[::nticks]
        fig.canvas.draw()
        ticklabels = [l.get_text() for l in ax1[ind].xaxis.get_ticklabels()][::nticks]
#        
        ax1[ind].xaxis.set_ticks(ticks)
        ax1[ind].xaxis.set_ticklabels(ticklabels)
        ax1[ind].figure.show()
        plt.ylabel('Q $m^3/s$')
        plt.title('Estación '+col)
        ax1[ind].set_ylim(bottom = 0)
    plt.legend(['Predictor','Original','Residual'],bbox_to_anchor=(1.05, 1), loc='upper left')    
    # plt.savefig(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Figuras\Rapel_Residuales_ABHN_1990_2004.png',dpi=300)
    Q_daily_MLR.to_csv(ruta_GitHub+'\Analisis-Oferta-Hidrica\Hidrología\Caudales\Validacion\Q_sin_canales_Q_relleno_MLR_Maule_'+str(year_i+1)+'-'+str(year_f)+'_outlier_in_correction_median.csv')

