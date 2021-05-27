# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 23:25:25 2020

@author: ccalvo
"""

import pandas as pd

def DD(texto_grados):
    D = float(texto_grados[0:2])
    M = float(texto_grados[4:6])
    S = float(texto_grados[8:10])
    dd = D+(M+S/60.)/60.
    return -dd

ruta = r'D:\GitHub\Analisis-Oferta-Hidrica\DGA\datosDGA\Pp\Mataquito\Estaciones_Mataquito_P_corregidas_DGA_v0.csv'

Q_Mataquito = pd.read_csv(ruta)
Q_Mataquito['Long'] = ''
Q_Mataquito['Lat'] = ''

for i in range(len(Q_Mataquito)):
    Q_Mataquito.loc[i,'Long'] = DD(Q_Mataquito.loc[i,'Longitud'])
    Q_Mataquito.loc[i,'Lat'] = DD(Q_Mataquito.loc[i,'Latitud'])
    
Q_Mataquito.to_csv('E:\CIREN\OneDrive - ciren.cl\Of hidrica\Clima\Q\Estaciones_Mataquito_Q_corregidas_DGA_v0.csv')

#%%

ruta = r'C:\Users\ccalvo\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\QAQC\Estaciones_Cuencas_Cabecera.csv'

est_CC = pd.read_csv(ruta, sep = ';')
est_CC['Long'] = ''
est_CC['Lat'] = ''

for i in range(len(est_CC)):
    est_CC.loc[i,'Long'] = DD(est_CC.loc[i,'Longitud'])
    est_CC.loc[i,'Lat'] = DD(est_CC.loc[i,'Latitud'])
    
est_CC.to_csv(r'C:\Users\ccalvo\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\QAQC\Estaciones_Cuencas_Cabecera_dd.csv')
