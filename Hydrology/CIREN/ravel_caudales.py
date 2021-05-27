# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 09:56:45 2021

@author: Carlos
"""

#Preámbulo
import pandas as pd

ruta = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos\Entrega_El_Planchon.xlsx'
ruta = r'C:\Users\Carlos\Downloads\AES GENER - 2008 - Adenda Proyecto Hidroeléctrico Alto Maipo Informe Final. Realizado por CONIC-BF-annotated.xlsx'


def ravel(ruta):
    q_Colorado_Maipo = pd.read_excel(ruta).iloc[28:-1,1:]
    q_Colorado_Maipo = q_Colorado_Maipo.values.ravel()
    
    
    q_El_Planchon = pd.read_excel(ruta).iloc[28:61,1:6]
    q_El_Planchon[q_El_Planchon < 0] = -q_El_Planchon[q_El_Planchon < 0]
    q_El_Planchon_ravel = q_El_Planchon.values.ravel()
