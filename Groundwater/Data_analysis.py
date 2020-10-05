# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 20:45:26 2020

@author: fcidm
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from subprocess import check_output
from matplotlib.ticker import FuncFormatter

import locale
# Set to German locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "de_DE")

import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()

# Tell matplotlib to use the locale we set above
plt.rcParams['axes.formatter.use_locale'] = True

plt.close("all")

ruta = r'E:\CORFO\Hidroquimica\BBDD_Hidroquímica.csv'
ruta2 = r'E:\CORFO\Hidroquimica\BBDD_isotopos.csv'
ruta_excel = r'E:\CORFO\Hidroquimica\BBDD_Hidroquímica.xlsx'

data = pd.read_csv(ruta, encoding = "ISO-8859-1")
data = pd.read_excel(ruta_excel, encoding = "ISO-8859-1")
data_isot = pd.read_csv(ruta2, encoding = "ISO-8859-1") 

data.head()

data.tail()

data.info()

data_superficial = data[data['Tipo de punto'] == 'Superficial']
data_subte = data[data['Tipo de punto'] == 'Subterráneo']


descr_superficial = data_superficial.describe()
descr_subte = data_subte.describe()

descr_superficial.to_csv('descr_subperficial.csv')
descr_subte.to_csv('descr_subterranea.csv')

descr = data.describe()
descr_isot = data_isot.describe()

data.fillna(value=np.nan, inplace=True)
data_isot.fillna(value=np.nan, inplace=True)


del data['Nombre']
#del data['Fecha muestreo']

data = data.replace('-', np.nan)
data = data.replace('ND', np.nan)
data = data.replace('>0', np.nan)
data = data.replace('NHP', np.nan)
data = data.replace('2,7', 2.7)
data = data.replace(' 0..5', .5)
data = data.replace('*', np.nan)
data = data.replace('.0.05', .5)
data.fillna(0)
data_isot.fillna(0)

data['Fecha muestreo'] = pd.to_datetime(data['Fecha muestreo'])
anios = data['Fecha muestreo'].dt.year
del data['Fecha muestreo']
del data['?13CDIC']
del data['?13CCO2']
del data['34SSO4']
del data['87Sr/86Sr']
del data['Indice SAR']
del data['DBO']
del data['Hidroxido']
del data['CO2']
del data['NH4+ disuelto mg/l']
del data['Ag disuelta mg/l']
del data['Be disuelto mg/l']
del data['Bi total mg/l']
del data['Cr disuelto mg/l']
del data['HBO3 mg/l']
#del data['Color verdadero']
del data['Clasificacion USSL']
del data['Detergente']
#del data['NO2']
del data['Ti total mg/l']
del data['Bi disuelto mg/l']
del data['Grasas y Aceites AyG mg/l']
del data['DQO']

borrar = ['Ba total mg/l','Br total mg/l','Co total mg/l','Cr total mg/l','Re disuelto mg/l','Sb total mg/l','V disuelto mg/l','Tl total mg/l','U total mg/l','U disuelto mg/l','V total mg/l', '2H','Compuestos fenoles','Color verdarero','Turbidez','Indice de Langelier','Br- disuelto mg/l','Cr_VI_disuelto mg/l','Sb disuelto mg/l','CN- disuelto mg/l','Co disuelto mg/l','Cs disuelto mg/l']
borrar2 = ['Re total mg/l','Sn disuelto mg/l','Ni total mg/l','NH3 disuelto mg/l','Si total mg/l','Alcalinidad carbonato mg/l','Be total mg/l','NO2-','Cu total mg/l','Cs total mg/l','Ag total mg/l','Tl disuelto mg/l','Carbono orgánico total mg/l','Al disuelto mg/l','Dureza (Ca, Mg) mg/l','I disuelto mg/l','Rb disuelto mg(l','Se total mg/l','Rb total mg/l','Si disuelto mg/l','Cd total mg/l','P-ortofosfato','F total mg/l','Alcalinidad total en meq/l','CN total mg/l','Eh mvol','Dureza (no carbonatos) mg/l']
borrar3 = ['Nitrogeno Amoniacal (N-NH4) mg/l','B total mg/l','S disuelto mg/l','Salinidad mg NaCl/l','Oxígeno disuelto mg/l','Cr_III_disuelto mg/l','Ti disuelto mg/l']
borrar = borrar + borrar2+borrar3
for label in borrar:
    del data[label]


data_isot['Fecha muestreo'] = pd.to_datetime(data_isot['Fecha muestreo'])
anios_i = data_isot['Fecha muestreo'].dt.year
anios_i = anios_i.astype(int)
del data_isot['Fecha muestreo']
del data_isot['Unnamed: 9']
del data_isot['Unnamed: 10']



data = data.notnull().astype('int')
data = data.groupby(anios)  
data_anual = data.aggregate(np.sum)
data_anual = data_anual[40:]/(365*0.8)  
data_anual = data_anual.apply(lambda x: [y if y < 1 else 1 for y in x])
data_anual = data_anual.transpose()

data_isot = data_isot.notnull().astype('int')
data_isot = data_isot.groupby(anios_i)  
data_isot_anual = data_isot.aggregate(np.sum)
data_isot_anual = data_isot_anual/(365*0.8)  
data_isot_anual = data_isot_anual.apply(lambda x: [y if y < 1 else 1 for y in x])
data_isot_anual = data_isot_anual.transpose()


f,ax = plt.subplots(figsize=(9, 9))
sns.heatmap(data_anual.astype(float), annot=False, linewidths=.5, fmt= '.1f',  cmap='YlGn')
comma_fmt = FuncFormatter(lambda x, p: format(int(x), ','))
plt.colorbar(format=comma_fmt)
plt.show()
plt.savefig('hidroquimica_anual.png')
plt.tight_layout()

f_i,ax_i = plt.subplots(figsize=(9, 9))
sns.heatmap(data_isot_anual.astype(float), annot=False, linewidths=.5, fmt= '.1f',  cmap='YlGn')
plt.show()
plt.savefig('isotopos_anual.png')