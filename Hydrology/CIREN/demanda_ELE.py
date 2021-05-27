# -*- coding: utf-8 -*-
"""
Created on Mon May 10 03:34:22 2021

@author: Carlos
"""

# librerias
# Importar librerias
from IPython.core.display import display, HTML
import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import modules_FAA
import modules_CCC
import geopandas
import contextily as ctx
import matplotlib.image as mpimg
import matplotlib
import locale
# Set to Spanish locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "es_ES")
locale.setlocale(locale.LC_TIME, "es_ES") # swedish
plt.rcdefaults()

# Tell matplotlib to use the locale we set above
plt.rcParams['axes.formatter.use_locale'] = True

# Paths y carga de datos genericos
path = '../Etapa 1 y 2/GIS/Cuencas_DARH/Cuencas/Cuencas_DARH_2015.shp'
basin = geopandas.read_file(path)
path = '../Etapa 1 y 2/GIS/Cuencas_DARH/Subcuencas/SubCuencas_DARH_2015.shp'
subbasin = geopandas.read_file(path)
ruta_Git = r'C:\Users\ccalvo\Documents\GitHub'
ruta_OD = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos'

    
# grafico de centrales
plt.close("all")
fig = plt.figure(figsize=(10, 8))
axes = fig.add_subplot(111)
ht = geopandas.read_file('..//SIG//Coberturas_Entrega//Hidroelectricas_Chile.shp')
te = geopandas.read_file('..//SIG//Coberturas_Entrega//Termoelectricas_Chile.shp')
# est_fluv = geopandas.read_file('E:\CIREN\OneDrive - ciren.cl\Of hidrica\GIS\estaciones_DGA.shp')

cuencas = geopandas.read_file('..//Etapa 1 y 2//GIS//Cuencas_DARH//Cuencas//Cuencas_DARH_2015.shp')
cuencas = cuencas.loc[(cuencas['COD_CUENCA'] == '0703') | (cuencas['COD_CUENCA'] == '0701') | (cuencas['COD_CUENCA'] == '0600') | (cuencas['COD_CUENCA'] == '1300')]
cuencas.plot(ax = axes, alpha=0.4)
# est_fluv.plot(ax = axes, color = 'cyan')
ht.plot(ax = axes, color = 'blue')
te.plot(ax = axes, color = 'red')
plt.legend(['Central Hidroeléctrica','Central Termoeléctrica'], loc='upper left')
ctx.add_basemap(ax = axes, crs= ht.crs.to_string(),
                        source = ctx.providers.Esri.WorldTerrain, zoom = 8)

# axes.legend(['Centros acuícolas en tierra'], loc = 'upper left')
x, y, arrow_length = 0.95, 0.95, 0.07
axes.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20,
            xycoords=axes.transAxes)
x, y, scale_len = cuencas.bounds['minx'], cuencas.bounds['miny'].min(), 20000 #arrowstyle='-'
scale_rect = matplotlib.patches.Rectangle((x.iloc[0],y),scale_len,200,linewidth=1,
                                        edgecolor='k',facecolor='k')
axes.add_patch(scale_rect)
plt.text(x.iloc[0]+scale_len/2, y+5000, s='20 KM', fontsize=10,
             horizontalalignment='center')
axes.set_xlabel('Coordenada Este WGS84 19S (m)')
axes.set_ylabel('Coordenada Norte WGS84 19S (m)')


#%%
# graficar demanda
##################################
### Centrales H. R. Metropolitana ###
##################################
dicc_RM = {
    
    'ALFALFAL' :    [0.91,	720],
    'Auxiliar Maipo'    :    [0.92,	27],
    'Carena'    :	  [0.92,	127],
    'El Llano'    :    [0.85,	34.4],
    'El Rincón'    :    [0.85,	68],
    'FLORIDA'    :   [0.92,	99],
    'Florida 2'    :   [0.92,	96],
    'Florida 3'    :    [0.92,	68],
    'Guayacán'    :   [0.92,	35],
    'Eyzaguirre'	:  [0.85,	22],
    'Las Vertientes'	:  [0.92,	27.8],
    'Los Bajos'	:   [0.92,	27],
    'LOS MORROS'	:  [0.92,	13],
    'MAITENES'	:    [0.92,	180],
    'Mallarauco'	:  [0.92,	100],
    'EPSA'	:    [0.92,	89.8], # Esta coirresponde a Puntilla con otro nombre
    'QUELTEHUES'	:  [0.91,	213],
    'VOLCAN'	:  [0.91,	181],
    'El Arrayán' : [.835, 73.5]}
    
    ###########################
    ### Centrales Rapel ###
    ###########################
 
dicc_Rapel = {
'Chacayes'	: [0.92,	181],
'Coya'	: [0.92,	137],
'El Paso'	: [0.91,	469],
'Confluencia'	: [0.92,	348],
'La Higuera'	: [0.92,	382],
'RAPEL'	: [0.92,	76],
'San Andrés'	: [0.91,	467],
'SAUZAL'	: [0.92,	118],
'SAUZALITO' :	[0.95,	25],
'Convento Viejo' : [.95, 23], #Turbinas Kaplan Fuente: EIA
'Dos Valles' : [.91, 390.8], #Turbinas Pelton Fuente: EIA
#    'Pangal' : [0.91, 448.], #Turbinas Pelton https://www.u-cursos.cl/ingenieria/2009/2/EL6000/1/material_docente/bajar?id_material=253406
'ech-la-compania-ii' : [.92, 43]} #Turbinas Francis, Fuente: DIA

###########################
### Centrales Mataquito ###
###########################   
dicc_Mataquito = {
'La Montaña 1' : [.91, 265],
'La Montaña 2' : [.91, 265]}
    
#######################
### Centrales Maule ###
#######################    

dicc_Maule = {
'Chiburgo'	: [0.92,	120],
'CIPRESES'  :    [0.91,  370], #Turbina Pelton de eje horizontal
'COLBUN'	:  [0.92,  168],
'Cumpeo'	:  [0.9667,	96],
'CURILLINQUE'	:  [0.92,	114.3],
'ISLA'	: [0.92,	93],
'Lircay'	: [0.92,	100],
'LOMA ALTA'	: [0.92,	50.4],
'Los Hierros'	: [0.9667,	103.2],
'Los Hierros II'	: [0.9667,	24.57],
'MACHICURA'	: [0.95,	37],
'Mariposas'	: [0.95,	35],
'Ojos de Agua'	: [0.92,	75],
'PEHUENCHE'	: [0.92,	206],
'Providencia'	: [0.92,	54],
'Purísima'	: [0.95,	9.3],
'Río Colorado' : [0.92 ,168.7],
'Robleria'	: [0.92,	125],    
'SAN CLEMENTE'	: [0.95,	35.5],
'SAN IGNACIO' :	[0.95,	21],
'Embalse Ancoa' : [0.92,  72], #Turbinas Francis Fuente: EIA
'Hidro La Mina' : [0.92,  61.58], #Turbinas Francis Fuente: EIA y https://snifa.sma.gob.cl/v2/General/DescargarInformeSeguimiento/46841
'El Galpón' : [.835 ,35.], #Turbina Ossberger 
}

dicc_vol_Maipo = {    
        
##################################
### Centrales R. Metropolitana ###
##################################

'CMPC Cordillera' : [13.7, 0.7], 'CMPC Tissue' : [13.7, 0.7],
'El Nogal' : [24.3, 1.2], 'Estancilla' : [24.3, 1.2], 'RENCA' : [27.65,1.4], 
'NUEVA RENCA' : [13.7, 0.7], 'Nueva Renca Diesel' : [24.3, 1.2], 'Nueva Renca GNL' : [13.7, 0.7], 'Nueva Renca FA_GLP' : [13.7, 0.7], 
'Chorrillos' : [24.3, 1.2], 'Sepultura' : [24.3, 1.2], 'Ermitaño' : [24.3, 1.2], 'Loma Los Colorados' : [31., 1.6], 
'Loma Los Colorados II' : [31., 1.6], 'Santa Marta' : [31., 1.6], 'Trebal Mapocho' : [31., 1.6], 
'El Campesino 1' : [31., 1.6]}
#edeuco' : [24.3,1.2],'

###########################
### Centrales O'Higgins ###
###########################

dicc_vol_Rapel = {
'Candelaria' : [13.7, 0.7], 'Candelaria 1' : [24.3, 1.2], 'Candelaria 2' : [24.3, 1.2], 'Candelaria Diesel' : [24.3, 1.2], 
'Candelaria GNL' : [13.7, 0.7], 'Esperanza' : [24.3,1.2], 'COLIHUES' : [24.3,1.2], 'Energía Pacífico' : [31., 1.6], 
'Las Pampas' : [31., 1.6], 'Santa Irene' : [31., 1.6], 'Tamm' : [31., 1.6]}

###########################
### Centrales Mataquito ###
###########################    

dicc_vol_Mataquito = {
  
'Teno' : [24.3,1.2], 'Zapallar' : [24.3,1.2], 'Cem Bio Bio IFO' : [24.3,1.2], 'Cem Bio Bio DIESEL' : [24.3,1.2], 'Cem Bio Bio' : [24.3,1.2],    
 }

#######################
### Centrales Maule ###
#######################    

dicc_vol_Maule = {
'San Lorenzo' : [24.3,1.2],  'Chile Generacion' : [24.3,1.2], 'Constitución 1' : [24.3,1.2], 
'Constitución 2' : [24.3,1.2], 'Raso Power' : [24.3,1.2], 'Licantén' : [31., 1.6], 
'Licantén LN' : [31., 1.6], 'Viñales' : [31., 1.6], 'CELCO': [31., 1.6], 'San Gregorio-Linares' : [24.3, 1.2], 'Maule' : [24.3, 1.2],

}
        
hidro = pd.read_csv(ruta_OD + r'\caudales_generacion_hidro.csv', parse_dates = True, index_col = 0)
hidro.index.names = ['']

termo = pd.read_csv(ruta_OD + r'\caudales_generacion_termo.csv', parse_dates = True, index_col = 0)
termo.index.names = ['']

# Maipo
hidro_Maipo = hidro[dicc_RM.keys()].sum(axis = 1)
# hidro_Maipo.index = hidro_Maipo.index.month
# hidro_2019 = hidro_2019.reindex([4,5,6,7,8,9,10,11,12,1,2,3]).reset_index(drop=True)
# hidro_2019.index =  ['Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct',
#                      'Nov', 'Dic', 'Ene', 'Feb', 'Mar']

#fig, ax = plt.subplots(1, figsize = (10,5))
# hidro_Maipo = hidro_2019.sum(axis = 1)
#plt.legend(title='Central hidroeléctrica', bbox_to_anchor=(1.05, 1.1), loc='upper left')
#plt.title('Caudal de uso hidroeléctrico cuenca del Río Maipo 2019')
#plt.ylabel('Q ($m^3$/s)')

termo_Maipo = termo[dicc_vol_Maipo.keys()].sum(axis = 1)
# termo_2019.index = termo_2019.index.month
# termo_2019 = termo_2019.reindex([4,5,6,7,8,9,10,11,12,1,2,3]).reset_index(drop=True)
# termo_2019.index =  ['Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct',
#                      'Nov', 'Dic', 'Ene', 'Feb', 'Mar']

#fig, ax = plt.subplots(1, figsize = (10,5))
# termo_Maipo = termo_2019.sum(axis = 1)
#plt.legend(title='Central termoeléctrica', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.title('Caudal de uso termoeléctrico cuenca del Río Maipo 2019')
#plt.ylabel('Q ($m^3$/s)')

# Rapel
hidro_Rapel = hidro[dicc_Rapel.keys()].sum(axis = 1)
# hidro_2019.index = hidro_2019.index.month
# hidro_2019 = hidro_2019.reindex([4,5,6,7,8,9,10,11,12,1,2,3]).reset_index(drop=True)
# hidro_2019.index =  ['Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct',
#                      'Nov', 'Dic', 'Ene', 'Feb', 'Mar']
#fig, ax = plt.subplots(1, figsize = (10,5))
# hidro_Rapel = hidro_2019.sum(axis = 1)
#plt.legend(title='Central hidroeléctrica', bbox_to_anchor=(1.05, 1.1), loc='upper left')
#plt.title('Caudal de uso hidroeléctrico cuenca del Río Rapel 2019')
#plt.ylabel('Q ($m^3$/s)')

termo_Rapel = termo[dicc_vol_Rapel.keys()].sum(axis = 1)
# termo_2019.index = termo_2019.index.month
# termo_2019 = termo_2019.reindex([4,5,6,7,8,9,10,11,12,1,2,3]).reset_index(drop=True)
# termo_2019.index =  ['Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct',
#                      'Nov', 'Dic', 'Ene', 'Feb', 'Mar']

#fig, ax = plt.subplots(1, figsize = (10,5))
# termo_Rapel = termo_2019.sum(axis = 1)
#plt.legend(title='Central termoeléctrica', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.title('Caudal de uso termoeléctrico cuenca del Río Rapel 2019')
#plt.ylabel('Q ($m^3$/s)')

# Mataquito
hidro_Mataquito = hidro[dicc_Mataquito.keys()].sum(axis = 1)
#plt.legend(title='Central hidroeléctrica', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.ylabel('Q ($m^3$/s)')

termo_Mataquito = termo[dicc_vol_Mataquito.keys()].sum(axis = 1)

#termo[dicc_vol_Mataquito.keys()].plot()
#plt.legend(title='Central termoeléctrica', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.ylabel('Q ($m^3$/s)')


#fig, ax = plt.subplots(1, figsize = (10,5))
#plt.legend(title='Central hidroeléctrica', bbox_to_anchor=(1.05, 1.1), loc='upper left')
#plt.title('Caudal de uso hidroeléctrico cuenca del Río Mataquito 2019')
#plt.ylabel('Q ($m^3$/s)')

#fig, ax = plt.subplots(1, figsize = (10,5))
#plt.legend(title='Central termoeléctrica', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.title('Caudal de uso termoeléctrico cuenca del Río Rapel 2019')
#plt.ylabel('Q ($m^3$/s)')

# Maule y graficos de demanda
#hidro[dicc_Maule.keys()].plot()
#plt.legend(title='Central hidroeléctrica', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.ylabel('Q ($m^3$/s)')

#termo[dicc_vol_Maule.keys()].plot()
#plt.legend(title='Central termoeléctrica', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.ylabel('Q ($m^3$/s)')

hidro_Maule = hidro[dicc_Maule.keys()].sum(axis = 1)
# hidro_2019.index = hidro_2019.index.month
# hidro_2019 = hidro_2019.reindex([4,5,6,7,8,9,10,11,12,1,2,3]).reset_index(drop=True)
# hidro_2019.index =  ['Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct',
#                      'Nov', 'Dic', 'Ene', 'Feb', 'Mar']
#fig, ax = plt.subplots(1, figsize = (10,5))
# hidro_Maule = hidro_2019.sum(axis = 1)
#plt.legend(title='Central hidroeléctrica', bbox_to_anchor=(1.05, 1.1), loc='upper left')
#plt.title('Caudal de uso hidroeléctrico cuenca del Río Mataquito 2019')
#plt.ylabel('Q ($m^3$/s)')

termo_Maule = termo[dicc_vol_Maule.keys()].sum(axis = 1)
# termo_2019.index = termo_2019.index.month
# termo_2019 = termo_2019.reindex([4,5,6,7,8,9,10,11,12,1,2,3]).reset_index(drop=True)
# termo_2019.index =  ['Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct',
#                      'Nov', 'Dic', 'Ene', 'Feb', 'Mar']

#fig, ax = plt.subplots(1, figsize = (10,5))
# termo_Maule = termo_2019.sum(axis = 1)
#plt.legend(title='Central termoeléctrica', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.title('Caudal de uso termoeléctrico cuenca del Río Rapel 2019')
#plt.ylabel('Q ($m^3$/s)')

hidro = {'Cuenca Río Maipo' : hidro_Maipo, 'Cuenca Río Rapel' : hidro_Rapel, 
              'Cuenca Río Mataquito' : hidro_Mataquito, 'Cuenca Río Maule' : hidro_Maule}

fig, ax = plt.subplots(2,2, figsize = (10,22))
ax = ax.reshape(-1)

for ind, cuenca in enumerate(hidro.keys()):
    hidro[cuenca].plot(ax = ax[ind])
    ax[ind].set_ylabel('Demanda de hidroeléctricas ($m^3/s$)')
    ax[ind].set_title(cuenca)
    ax[ind].set_ylim(bottom = 0)
    if cuenca == 'Cuenca Río Mataquito':
        ax[ind].set_ylim(top = 2)
    else:
        ax[ind].set_ylim(top = 1400)
    ax[ind].grid()


termo = {'Cuenca Río Maipo' : termo_Maipo, 'Cuenca Río Rapel' : termo_Rapel, 'Cuenca Río Mataquito' : termo_Mataquito,
             'Cuenca Río Maule' : termo_Maule}

fig, ax = plt.subplots(2,2, figsize = (10,22))
ax = ax.reshape(-1)

import locale
# Set to Spanish locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "es_ES")
locale.setlocale(locale.LC_TIME, "es_ES") # swedish

for ind, cuenca in enumerate(termo.keys()):
    termo[cuenca].plot(ax = ax[ind])
    ax[ind].set_ylabel('Demanda de termoeléctricas ($m^3/s$)')
    ax[ind].set_title(cuenca)
    ax[ind].set_ylim(bottom = 0)
    ax[ind].set_ylim(top = 0.11)
    ax[ind].grid()

    #ax[ind].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))