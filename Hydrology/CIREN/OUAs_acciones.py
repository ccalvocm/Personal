# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 08:47:25 2021

@author: Carlos
"""

#===================================
#             librerías
#===================================

import pandas as pd
import math
import numpy as np
from numba import jit
import requests
import zipfile
import os
import random
from time import sleep
import pandas as pd
import random
import numpy as np
import geopandas
import matplotlib.pyplot as plt
from unidecode import unidecode



#===================================
#               rutas 
#===================================

ruta_OUA = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Solicitudes de información\Recibidos\DGA\OU_Aprobadas_Regiones RM 6 y 7.xls'
ruta_OD = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC'
os.chdir(ruta_OD+r'\Scripts')
ruta_cuencas = '../Etapa 1 y 2/GIS/Cuencas_DARH/Cuencas/Cuencas_DARH_2015.shp'
basin = geopandas.read_file(ruta_cuencas)

def tabla(df):
    return df[['Código de Expediente','Nombre de Organización/ Persona','Acciones Fuente', 'Acciones Cauce','Cantidad_integrantes','Referencia a puntos conocidos I.1', 'Referencia a puntos conocidos II',
       'Referencia a puntos conocidos III', 'Referencia a puntos conocidos IV',
       'UTM \nNorte', 'UTM Este', 'Huso ', 'Datum', 'Latitud', 'Longitud',
       'Datum.1']]

def BDDDGA():
        
    #===================================
    # Agregar region al archivo de OUA 
    #===================================
    
    
    oua = pd.read_excel(ruta_OUA, skiprows = 6)
    region = ''
    for i, j in oua.iterrows(): 
        aux = oua.loc[i,'Región']
        if isinstance(aux, str):
            region = aux
        oua.loc[i,'Región'] = region
    
    #=======================================
    #        Región Metropolitana 
    #=======================================
    
    oua_RM = oua[oua['Región'] == 'Metropolitana']
    
    # ----------BDD DGA
    
    JDV_AC_CA_oua_RM = oua_RM[oua_RM['Provincia'].notna()]
    
    # ----------Integrantes
    
    integrantes_oua_RM = oua_RM[oua_RM['Comuna/ Tipo Persona'].isin(['INTEGRANTE'])]
    integrantes_oua_RM['Numero_integrante'] = 1
    integrantes_oua_RM = integrantes_oua_RM.groupby(['Código de Expediente']).sum()
    
    # ---------Juntas de Vigilancia
    
    jv_RM = JDV_AC_CA_oua_RM[JDV_AC_CA_oua_RM['Código de Expediente'].str.contains('NJ-') > 0]
    jv_RM['Cantidad_integrantes'] = integrantes_oua_RM.loc[jv_RM['Código de Expediente'],'Numero_integrante'].values
    print(tabla(jv_RM))
        
    # ---------Asociaciones de canalistas
    ac_RM = JDV_AC_CA_oua_RM[JDV_AC_CA_oua_RM['Código de Expediente'].str.contains('NA-') > 0]
    indice = ac_RM['Código de Expediente'][ac_RM['Código de Expediente'].isin(integrantes_oua_RM.index)]
    ac_RM.loc[indice.index,'Cantidad_integrantes'] = integrantes_oua_RM.loc[indice,'Numero_integrante'].values
    print(ac_RM[['Nombre de Organización/ Persona']])
    
    # ---------Comunidades de aguas
    
    ca_RM = JDV_AC_CA_oua_RM[JDV_AC_CA_oua_RM['Código de Expediente'].str.contains('NC-') > 0]
    indice = ca_RM['Código de Expediente'][ca_RM['Código de Expediente'].isin(integrantes_oua_RM.index)]
    ca_RM.loc[indice.index,'Cantidad_integrantes'] = integrantes_oua_RM.loc[indice,'Numero_integrante'].values
    print(tabla(ca_RM))
    
    writer = pd.ExcelWriter('..\\scripts\\outputs\\OUAs\\OUAs.xlsx', engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    tabla(jv_RM).to_excel(writer, sheet_name='JDV_RM', index=False)
    tabla(ac_RM).to_excel(writer, sheet_name='AC_RM', index=False)
    tabla(ca_RM).to_excel(writer, sheet_name='CA_RM', index=False)
        

    #=============================
    #            O'Higgins  
    #=============================
    
    oua_OH = oua[oua['Región'] == 'OHiggins']
    
    # ----------BDD DGA
    
    JDV_AC_CA_oua_OH = oua_OH[oua_OH['Provincia'].notna()]
    
    # ----------Integrantes
    
    integrantes_oua_OH = oua_OH[oua_OH['Comuna/ Tipo Persona'].isin(['INTEGRANTE'])]
    integrantes_oua_OH['Numero_integrante'] = 1
    integrantes_oua_OH = integrantes_oua_OH.groupby(['Código de Expediente']).sum()
    
    # ---------Juntas de Vigilancia
    
    jv_OH = JDV_AC_CA_oua_OH[JDV_AC_CA_oua_OH['Código de Expediente'].str.contains('NJ-') > 0]
    indice = jv_OH['Código de Expediente'][jv_OH['Código de Expediente'].isin(integrantes_oua_OH.index)]
    jv_OH.loc[indice.index,'Cantidad_integrantes'] = integrantes_oua_OH.loc[indice,'Numero_integrante'].values
    print(jv_OH[['Nombre de Organización/ Persona']])
    
    # ---------Asociaciones de canalistas
    ac_OH = JDV_AC_CA_oua_OH[JDV_AC_CA_oua_OH['Código de Expediente'].str.contains('NA-') > 0]
    indice = ac_OH['Código de Expediente'][ac_OH['Código de Expediente'].isin(integrantes_oua_OH.index)]
    ac_OH.loc[indice.index,'Cantidad_integrantes'] = integrantes_oua_OH.loc[indice,'Numero_integrante'].values
    print(ac_OH[['Nombre de Organización/ Persona']])
    
    # ---------Comunidades de aguas
    
    ca_OH = JDV_AC_CA_oua_OH[JDV_AC_CA_oua_OH['Código de Expediente'].str.contains('NC-') > 0]
    indice = ca_OH['Código de Expediente'][ca_OH['Código de Expediente'].isin(integrantes_oua_OH.index)]
    ca_OH.loc[indice.index,'Cantidad_integrantes'] = integrantes_oua_OH.loc[indice,'Numero_integrante'].values
    print(ca_OH[['Nombre de Organización/ Persona']])
    
    tabla(jv_OH).to_excel(writer, sheet_name='JDV_OH', index=False)
    tabla(ac_OH).to_excel(writer, sheet_name='AC_OH', index=False)
    tabla(ca_OH).to_excel(writer, sheet_name='CA_OH', index=False)
        
    #=============================
    #            Maule  
    #=============================
    
    
    oua_Maule = oua[oua['Región'] == 'Maule']   
    
    # ----------BDD DGA
    
    JDV_AC_CA_oua_Maule = oua_Maule[oua_Maule['Provincia'].notna()]
    
    # ----------Integrantes
    
    integrantes_oua_Maule = oua_Maule[oua_Maule['Comuna/ Tipo Persona'].isin(['INTEGRANTE'])]
    integrantes_oua_Maule['Numero_integrante'] = 1
    integrantes_oua_Maule = integrantes_oua_Maule.groupby(['Código de Expediente']).sum()

    # ---------Juntas de Vigilancia
    
    jv_Maule = JDV_AC_CA_oua_Maule[JDV_AC_CA_oua_Maule['Código de Expediente'].str.contains('NJ-') > 0]
    indice = jv_Maule['Código de Expediente'][jv_Maule['Código de Expediente'].isin(integrantes_oua_Maule.index)]
    jv_Maule.loc[indice.index,'Cantidad_integrantes'] = integrantes_oua_Maule.loc[indice,'Numero_integrante'].values
    print(jv_Maule[['Nombre de Organización/ Persona']])
    
    # ---------Asociaciones de canalistas
    ac_Maule = JDV_AC_CA_oua_Maule[JDV_AC_CA_oua_Maule['Código de Expediente'].str.contains('NA-') > 0]
    indice = ac_Maule['Código de Expediente'][ac_Maule['Código de Expediente'].isin(integrantes_oua_Maule.index)]
    ac_Maule.loc[indice.index,'Cantidad_integrantes'] = integrantes_oua_Maule.loc[indice,'Numero_integrante'].values
    print(ac_Maule[['Nombre de Organización/ Persona']])
    
    # ---------Comunidades de aguas
    
    ca_Maule = JDV_AC_CA_oua_Maule[JDV_AC_CA_oua_Maule['Código de Expediente'].str.contains('NC-') > 0]
    indice = ca_Maule['Código de Expediente'][ca_Maule['Código de Expediente'].isin(integrantes_oua_Maule.index)]
    ca_Maule.loc[indice.index,'Cantidad_integrantes'] = integrantes_oua_Maule.loc[indice,'Numero_integrante'].values
    print(ca_Maule[['Nombre de Organización/ Persona']])
    
    
    tabla(jv_Maule).to_excel(writer, sheet_name='JDV_MAULE', index=False)
    tabla(ac_Maule).to_excel(writer, sheet_name='AC_MAULE', index=False)
    tabla(ca_Maule).to_excel(writer, sheet_name='CA_MAULE', index=False)
    
        
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    writer.close()

    # =========================================
    # shp canales
    # -----------------------------------------
    # Ahora cruzar el shp de canales con la BDD
    # -----------------------------------------
    
    geodf_cMaule = geopandas.read_file(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\SIG\AC_CA_Maule\canales_maule_2020.shp')
    geodf_cMaule['OBJECTID_1']
    df_cMaule = pd.read_excel(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\SIG\AC_CA_Maule\canales_maule_2020_26_abril.xlsx', sheet_name = 'canales_CA_AC')
    df_cMaule.index = df_cMaule['OBJECTID_1']
    df_cMaule_2 = df_cMaule.copy()[df_cMaule.columns[25:]]
    geodf_merge = pd.merge(geodf_cMaule, df_cMaule_2, on='OBJECTID_1')

    # --------------------------------
    # formatear Fechas
    # --------------------------------
    lista_campos = ['Fecha \nResolución','Fecha Sentencia']
    
    for campo in lista_campos:
        geodf_merge[campo] = geodf_merge[campo].astype(str)
    
    # --------------------------------
    # formatear enteros
    # --------------------------------
    
    lista_enteros = ['Nº \nResolución ','N° CBR','Año','N° Registro en Libro D.L.','Datum','Datum.1']

    for campoentero in lista_enteros:
        geodf_merge[campoentero] = geodf_merge[campoentero].astype(pd.Int32Dtype())   
        geodf_merge[campoentero] = geodf_merge[campoentero].astype(str).replace('<NA>','')    
    
    # --------------------------------
    # decodificar columnas
    # --------------------------------
    
    geodf_merge.columns = [unidecode(x) for x in geodf_merge.columns]
    geodf_merge.to_file("E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\SIG\AC_CA_Maule\canales_Maule_2020_merge.shp",
                        encoding = 'utf-8')


def BDD_CIREN():
    
     # ----------BDD CIREN
    
    jur_jv_CIREN = geopandas.read_file('..\\..\\SIG\\OUAs\\Region_VI\\JV_Jurisdiccion_VI.shp')    
    jur_jv_CIREN.plot()
    
       
    # ----------BDD CIREN
    
    jur_jv_CIREN = geopandas.read_file('..\\..\\SIG\\OUAs\\Region_VI\\JV_Jurisdiccion_VI.shp')    
    jur_jv_CIREN.plot()
    
    
    # cuencas 
    
    
    cuenca_Mataquito = basin[basin['COD_CUENCA'] == '0701']
    fig, ax = plt.subplots(1)
    
    
    # zonas de riego 
    
    
    ZR_Maule = geopandas.read_file(ruta_OD+r'\SIG\COBERTURAS INFRAESTRUCTURA Y ÁREAS DE RIEGO\Infraestructura_Riego_Maule_2020\AR_Maule_2020.shp')
    
    ZR_Mataquito = geopandas.clip(cuenca_Mataquito, ZR_Maule,  keep_geom_type=False)
    ZR_Mataquito.plot(ax = ax)
    cuenca_Mataquito.plot(ax = ax, alpha = 0.3)
    
    canales_ZR = ZR_Mataquito['NOM_CAN_MA'][ZR_Mataquito['NOM_CAN_MA'].str.contains('LA HUERTA') > 0]



