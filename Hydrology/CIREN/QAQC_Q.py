# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 22:39:56 2021

@author: Carlos
"""

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from matplotlib.ticker import FuncFormatter
import locale
import csv
from hydrobox import toolbox
import scipy.stats as st
import statsmodels as sm
import inspect
from matplotlib.font_manager import FontProperties
from itertools import cycle


def digito_verificador(rut):
    reversed_digits = map(int, reversed(str(rut)))
    factors = cycle(range(2, 8))
    s = sum(d * f for d, f in zip(reversed_digits, factors))
    if (-s) % 11 > 9:
        return 'K'
    else:
        return (-s) % 11
    
def main():
    
    # Maule
    
    root = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\QAQC\ValidacionCF'
    root_David = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos\Datos CF\Estaciones\Estaciones\Series variables hidrometeorológicas\DGA\Q\Q_mensual_DGA_1989_2019.csv'
    root_CR2 = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\Clima\Q\cr2_qflxAmon_2018\cr2_qflxAmon_2018.txt'
    os.chdir(root)
    Q_BNA = pd.read_excel(root+'\\Q_BNA.xlsx')
    Q_BNA.index =  pd.to_datetime([f'{y}-{m}-01' for y, m in zip(Q_BNA['anno'], Q_BNA['Mes'])])
    Q_david = pd.read_csv(root_David)
    Q_david.index =  pd.to_datetime([f'{y}-{m}-01' for y, m in zip(Q_david['anno'], Q_david['Mes'])])
    Q_cr2 = pd.read_csv(root_CR2, sep = ',', na_values=["-9999"], index_col = 0)
    for x in Q_cr2.columns:
        Q_cr2.rename(columns={x: '0'+str(x)+'-'+str(digito_verificador(x))}, inplace=True)    
    Q_cr2 = Q_cr2.iloc[14:]
    Q_cr2.index = pd.to_datetime(Q_cr2.index)
    
    inicio = pd.to_datetime('2008-01-01',format='%Y-%m-%d')
    fin = pd.to_datetime('2019-12-01',format='%Y-%m-%d')
  
    Q_david_2 = pd.DataFrame(Q_david[['Rio_loncomilla_en_las_brisas_DGA','Rio_maule_en_forel_DGA',
                                      'Rio_melado_en_el_salto_DGA', 'Rio_perquilauquen_en_quella_DGA']].loc[(Q_david.index <= fin) & (Q_david.index >= inicio)])
    
    Q_cr2_2 = pd.DataFrame(Q_cr2[Q_BNA.columns[2:]].loc[(Q_cr2.index <= fin) & (Q_cr2.index >= inicio)])
    Q_cr2_2.index.names = ['']
    
    
    flags = {'07359001-9' : ['2008-06-01', '2008-07-01' , '2010-08-01', '2010-09-01' ,'2016-08-01'],
             '07383001-K' : ['2011-01-01','2011-02-01','2014-10-01', '2015-06-01', '2016-08-01', '2017-04-01', '2017-05-01'],
             '07317005-2' : ['2014-01-01', '2014-02-01', '2016-08-01', '2016-10-01' ],
             '07335001-8' : ['2011-11-01', '2014-04-01', '2014-10-01', '2016-01-01', '2016-08-01', ]}
    
    for key in flags:
        for date in flags[key]:
            Q_BNA.loc[pd.to_datetime(date),key] = np.nan         
        
    plt.close("all")
    
    nticks = 2
    fig = plt.figure(figsize = (15,11))
    plt.suptitle('Serie de caudales BNA, David y CR2')
    for n,col in enumerate(Q_david_2.columns):
        fig.add_subplot(2,2,n+1)
        plt.title('Est. #' + col)
        ax1 = Q_BNA.iloc[:,n+2].plot(color = 'blue', linewidth = 2, label = 'BNA')
        Q_david_2[col].plot(color = 'red', linewidth = 1, label = 'David', ax = ax1)
        Q_cr2_2.iloc[:,n].plot(color = 'orange', linewidth = 1, label = 'David', ax = ax1)     
        
        ticks = ax1.xaxis.get_ticklocs()[::nticks]
        fig.canvas.draw()
        ticklabels = [l.get_text() for l in ax1.xaxis.get_ticklabels()][::nticks]
        
        ax1.xaxis.set_ticks(ticks)
        ax1.xaxis.set_ticklabels(ticklabels)
        ax1.figure.show()
        ax1.set_ylabel('Q ($m^3/s$)')
    
    plt.legend(['BNA', 'CF','CR2'], bbox_to_anchor=(1.05, 1), loc='upper left')

    # Maipo
    
    
if __name__ == '__main__':
    main()
    
    
    # fig = plt.figure(figsize = (15,12))
    # plt.suptitle('Histograma de diferencias de caudales DGA y CR2')
    # diferenciasQ = Q_DGA_benchmark-Q_CR2_benchmark
    # for n,col in enumerate(diferenciasQ.columns):
    #     fig.add_subplot(11,6,n+1)
    #     diferenciasQ[col].plot.hist()
    #     plt.xlim([-20,20])
    #     # plt.ylim([0,2])
    # plt.title(col)


   

