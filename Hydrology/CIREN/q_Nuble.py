# -*- coding: utf-8 -*-
"""
Created on Tue May 11 20:59:34 2021

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
import geopandas


def flags_mon(df):
    df_flag = df.copy()
    df_flag[:] = 1
    df_flag[df.isnull()] = 0
    df_flag = df_flag.resample('MS').sum()
    df_mon = df.copy().apply(pd.to_numeric).resample('MS').mean()[df_flag > 20]
    return df_mon

def digito_verificador(rut):
    reversed_digits = map(int, reversed(str(rut)))
    factors = cycle(range(2, 8))
    s = sum(d * f for d, f in zip(reversed_digits, factors))
    if (-s) % 11 > 9:
        return 'K'
    else:
        return (-s) % 11
    
def main():
    
    # leer caudales    
    root = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos\Datos CF\Datos_BNA_EstacionesDGA\BNAT_CaudalDiario.txt'
    Q_BNA = pd.read_csv(root, sep = ';', index_col = 0, names = ['name','date','q','flags'])
    
    # caudales ñuble
    q_nuble = Q_BNA.loc[Q_BNA.index.isin(['08105001-5','08106001-0','08106002-9','08105006-6'])]
    df_nuble = pd.DataFrame([], index = pd.date_range('1800-04-01','2020-03-31',freq = 'd'), columns = list(dict.fromkeys(q_nuble.index.to_list())))
    for est in df_nuble.columns:
        idx = pd.to_datetime(q_nuble.loc[est,'date'].values, dayfirst = True)
        df_nuble.loc[idx,est]  = q_nuble.loc[est,'q'].values
        
    # flags de meses con 20 dias minimos de informacion
    df_nuble = df_nuble.loc[(df_nuble.index >= '1979-04-01') & (df_nuble.index < '2020-04-01')]
    df_nuble_flags = flags_mon(df_nuble)
    
    # transposición de caudales
    nuble_sanfabian = geopandas.read_file(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\SIG\Ñuble\Cuencas\8106001.shp')
    nuble_sanfabian.set_crs(epsg = 32719)
    nuble_sanfabian_2 = geopandas.read_file(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\SIG\Ñuble\Cuencas\8106002.shp')
    nuble_sanfabian_2.set_crs(epsg = 32719)   
    
    # area san fabian 1 / area san fabian 2
    area_fabian1_2 = nuble_sanfabian.area/nuble_sanfabian_2.area
    
    idx = df_nuble_flags['08106001-0'][df_nuble_flags['08106001-0'].isnull()].index.intersection(df_nuble_flags['08106002-9'][df_nuble_flags['08106002-9'].notnull()].index)
    df_nuble_flags.loc[idx,'08106001-0'] = df_nuble_flags.loc[idx,'08106002-9']*area_fabian1_2.values[0]
    
    # area san fabian 2 / area san fabian 1
    idx = df_nuble_flags['08106002-9'][df_nuble_flags['08106002-9'].isnull()].index.intersection(df_nuble_flags['08106001-0'][df_nuble_flags['08106001-0'].notnull()].index)
    df_nuble_flags.loc[idx,'08106002-9'] = df_nuble_flags.loc[idx,'08106001-0']/area_fabian1_2.values[0]
    
    for mes in range(1,13):
        df_mes = df_nuble_flags.loc[df_nuble_flags.index.month == mes]
        for col in df_mes.columns:
            df_mes.loc[df_mes.index,col] = df_mes[col].fillna(df_mes[col].median())
        df_nuble_flags.loc[df_mes.index] = df_mes.values
    
    df_nuble_flags.to_csv(r'q_Ñuble_mon.csv')

if __name__ == '__main__':
    main()
