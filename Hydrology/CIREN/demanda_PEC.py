#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 12:13:07 2021

@author: faarrosp
"""
import os
import pandas as pd
from unidecode import unidecode
import geopandas as gpd
from IPython.display import display
import geopandas as gpd
from matplotlib import pyplot as plt
import contextily as ctx


#------------ Define folders and paths
folder_pec = os.path.join('..', 'Etapa 1 y 2', 'Demanda', 'PEC', 'Fuentes')
xls_ine_ave = os.path.join(folder_pec, '19Aves_mod.xls')
xls_ine_bov = os.path.join(folder_pec, '20Gan_Bovino_mod.xls')
xls_ine_cap = os.path.join(folder_pec, '21Gan_Caprino_mod.xls')
xls_ine_ovi = os.path.join(folder_pec, '22Gan_Ovino_mod.xls')
xls_ine_otr = os.path.join(folder_pec, '23Gan_Otras_Espec_mod.xls')

#------------ Create dataframe 'aves'
df_ine_ave = pd.read_excel(xls_ine_ave)
df_ine_ave.dropna(subset=['Comuna'], inplace=True)
df_ine_ave.rename({'Gallos Gallinas Pollos y Pollas': 'Broilers'}, axis=1, inplace=True)
df_ine_ave = df_ine_ave.melt(id_vars=['Comuna'], var_name='especie', value_name='cabezas')
df_ine_ave['ganado'] = 'ave'

#------------ Create dataframe 'bovinos'
df_ine_bov = pd.read_excel(xls_ine_bov)
df_ine_bov.dropna(subset=['Comuna'], inplace=True)
df_ine_bov = df_ine_bov.melt(id_vars=['Comuna'], var_name='especie', value_name='cabezas')
df_ine_bov['ganado'] = 'bov'
df_ine_bov.head()

#------------ Create dataframe 'caprinos'
df_ine_cap = pd.read_excel(xls_ine_cap)
df_ine_cap.dropna(subset=['Comuna'], inplace=True)
df_ine_cap = df_ine_cap.melt(id_vars=['Comuna'], var_name='especie', value_name='cabezas')
df_ine_cap['ganado'] = 'cap'
df_ine_cap.head()

#------------ Create dataframe 'ovinos'
df_ine_ovi = pd.read_excel(xls_ine_ovi)
df_ine_ovi.dropna(subset=['Comuna'], inplace=True)
df_ine_ovi = df_ine_ovi.melt(id_vars=['Comuna'], var_name='especie', value_name='cabezas')
df_ine_ovi['ganado'] = 'ovi'
df_ine_ovi.head()

#------------ Create dataframe 'otros'
df_ine_otr = pd.read_excel(xls_ine_otr)
df_ine_otr.dropna(subset=['Comuna'], inplace=True)
df_ine_otr = df_ine_otr.melt(id_vars=['Comuna'], var_name='especie', value_name='cabezas')
df_ine_otr['ganado'] = 'otr'
df_ine_otr.head()

#%%
#----------- Merge all dataframes
df = pd.concat([df_ine_ave, df_ine_bov, df_ine_cap, df_ine_otr, df_ine_ovi], ignore_index=True)
df['Comuna'] = df['Comuna'].apply(unidecode).str.title()

#----------- Get the 'comunas' series from the merged INE dataframe
comunas_ine = df['Comuna'].drop_duplicates().str.title()
comunas_ine = comunas_ine.apply(unidecode).sort_values()

#----------- Get the 'comuna' code and 'region' code by cross-validating the 'Comuna' series from INE data and the 'chile_comupol_2019.shp' file

#----- import the chile_comupol shapefile
folder_comunas_shp = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Politico', 'chile_comupol_2019')
comunas_shp = os.path.join(folder_comunas_shp, 'chile_comupol_2019.shp')
gdf_comunas = gpd.read_file(comunas_shp)
gdf_comunas.drop(['id', 'shape_leng', 'shape_area', 'codpro', 'nom_pro', 'nom_reg'], axis=1, inplace=True)
comunas_shp = gdf_comunas.drop('geometry', axis=1).sort_values(by = 'nom_com')

#----- format the name of comunas
comunas_shp['nom_com'] = comunas_shp['nom_com'].apply(unidecode).str.title()

#----- cross validate the comunas series
comunas_ine_final = pd.DataFrame(comunas_ine) # create dataframe from series
comunas_ine_final.rename({'Comuna': 'nom_com'}, axis=1, inplace=True) # rename column label to cross validate
comunas_ine_final = comunas_ine_final.join(comunas_shp.set_index('nom_com'), on='nom_com') # JOIN both dataframes
comunas_ine_final.dropna(inplace=True) # drop the NA values ('Juan Fernandez' and 'Isla de Pascua')
# print(comunas_ine_final.isna().sum())

#----------- Consolidate changes on the merged dataframe
df.rename({'Comuna': 'nom_com'}, axis=1, inplace=True) # rename column label to cross validate
df = df.join(comunas_ine_final.set_index('nom_com'), on='nom_com')
display(df.head())

df_ine = df.copy()
del df

#%% --------------  data de ODEPA

#------------- Diccionario para meses
dicmes = {'enero': '1', 'febrero': '2', 'marzo': '3', 'abril': '4',
          'mayo': '5', 'junio': '6', 'julio': '7', 'agosto': '8',
          'septiembre': '9', 'octubre': '10', 'noviembre': '11',
          'diciembre': '12'}

# dicreg = {'Arica y Parinacota, Tarapaca y Antofagasta': ['15', '01', '02'],
#           'Biobio y La Araucania': ['08', '09'],
#           'Coquimbo y Valparaiso': ['04', '05'],
#           'Region Metropolitana de Santiago': ['13'],
#           'Region del Libertador Bernardo OHiggins': ['06'],
#           'Region de Arica y Parinacota': ['15'],
#           'Arica y Parinacota, Coquimbo y Valparaiso': ['15', '04', '05'],
#           'OHiggins, Nuble, Biobio y La Araucania': ['06', '16', '08', '09']}

dicreg = {'Region de ': '',
          'Arica y Parinacota': '15',
          'Tarapaca': '01',
          'Antofagasta': '02',
          'Atacama': '03',
          'Coquimbo': '04',
          'Valparaiso': '05',
          'Region Metropolitana de Santiago': '13',
          'Region del Libertador Bernardo OHiggins': '06',
          'Region del': '',
          'Maule': '07',
          'Biobio': '08',
          'La Araucania': '09',
          'Los Rios': '14',
          'Los Lagos': '10',
          'OHiggins': '06',
          'Nuble': '16',
          'Aisen del Gral Carlos Ibanez del Campo': '11',
          'Magallanes y de la Antartica Chilena': '12',
          'Region': ''}

#------------ Define folders and paths
folder_pec = os.path.join('..', 'Etapa 1 y 2', 'Demanda', 'PEC', 'Fuentes')
xls_odepa_ave = os.path.join(folder_pec, 'DetalleRegionMensual_202145_Ave.xls')
xls_odepa_bov = os.path.join(folder_pec, 'DetalleRegionMensual_202145_Bov.xls')
xls_odepa_otros = os.path.join(folder_pec, 'DetalleRegionMensual_202145_Otros.xls')

#------------ Create dataframes
df_odepa_ave = pd.read_excel(xls_odepa_ave, skipfooter=1) # otras aves incluye pato, ganso, avestruz, etc
df_odepa_bov = pd.read_excel(xls_odepa_bov, skipfooter=1)
df_odepa_otr = pd.read_excel(xls_odepa_otros, skipfooter=1)

df_odepa_ave.rename({'Otras aves *': 'Otros'}, axis=1, inplace=True)
#------------ apply unidecode for columns and data
# df_odepa_ave.columns = [unidecode(x) for x in df_odepa_ave.columns]
# df_odepa_ave['Region'] = df_odepa_ave['Region'].apply(unidecode)

def apply_unidecode_strreplace_lower(df):
    df.columns = [unidecode(x) for x in df.columns]
    df['Region'] = df['Region'].apply(unidecode)
    df['Region'] = df['Region'].str.replace("'", "")
    df['Region'] = df['Region'].str.replace(".", "")
    df['Mes'] = df['Mes'].str.lower()
    df['Mes'] = df['Mes'].replace(dicmes, regex=True)
    df['Region'] = df['Region'].replace(dicreg, regex=True)
    df['Region'] = df['Region'].str.replace('y', ',')
    df['Region'] = df['Region'].str.replace('-', ',')
    df['Region'] = df['Region'].str.replace(' ', '')
    
def drop_totals(df):
    df.drop('Total', axis=1, inplace=True)
    
def unpivot(dataframe):
    dataframe = dataframe.melt(id_vars=['Ano', 'Region', 'Mes'],
                 var_name='especie', value_name='cabezas')
    return dataframe


# funcion que toma un grupo de regiones de ODEPA y el grupo de especie y
# entrega el vector de ponderadores por el cual multiplicar las cantidades

def get_ponderadores(dfine,dfodepa, regstr, esp, option):
    filtro = dfine.isin({'grupoesp': [esp],
                          'codreg': regstr.split(',')})
    filtro = filtro.grupoesp & filtro.codreg
    serie = dfine[filtro].groupby(option).sum()['cabezas']
    cods = serie.index.values
    vector = (serie/serie.sum()).values
    return cods,vector

def get_new_odepa_df(ine,odepa,option):
    input_rows = []
    
    for idx, row in odepa.iterrows():
        regstr = row['Region']
        regs = regstr.split(',')
        esp = row['grupoesp']
        cab = row['cabezas']
        cods, ponds = get_ponderadores(ine, odepa, regstr, esp, option)
        
           
        for cod, pond in zip(cods, ponds):
            rownew = row.to_dict()
            rownew.update({option:cod, 'cabezas': cab * pond})
            input_rows.append(rownew)
            
    
    odepanew = pd.DataFrame(input_rows)  
    return odepanew

    

for dataframe in [df_odepa_ave, df_odepa_otr, df_odepa_bov]:
    apply_unidecode_strreplace_lower(dataframe)
    drop_totals(dataframe)
    
#%%%%%%%%%%%%%%% odepa AVES

aveine = df_ine[df_ine['ganado'].isin(['ave'])].copy()
aveodepa = unpivot(df_odepa_ave).copy()

for x in [aveine, aveodepa]:
    x['especie'] = x['especie'].apply(unidecode)

# primero debemos homologar las categorias
#   para ello definimos el diccionario que lo hara por nosotros

dicine = {'Broilers':1, 'Pavos':2, 'Patos':3, 'Gansos':3, 'Avestruces':3,
          'Emues':3, 'Codornices':3, 'Faisanes':3, 'Otras Aves':3}
aveine['grupoesp'] = aveine['especie'].replace(dicine,regex=True)

dicodepa = {'Broilers': 1, 'Gallinas':1, 'Pavos':2, 'Otros':3}
aveodepa['grupoesp'] = aveodepa['especie'].replace(dicodepa,regex=True)


aveodepanew = get_new_odepa_df(aveine, aveodepa, 'codcom')
aveodepanew['ganado'] = 'ave'
aveodepanew = aveodepanew.join(comunas_shp.set_index('codcom'), on='codcom')
avets = aveodepanew.pivot_table(columns = 'Ano',
                                values = 'cabezas', index='codreg',
                                aggfunc='sum')


#%% --------------- odepa BOVINOS
bovine = df_ine[df_ine['ganado'].isin(['bov'])].copy()
bovodepa = unpivot(df_odepa_bov).copy()

dicine = dict(zip(bovine['especie'].unique(), [3,4,2,6,6,5,1,1]))
dicodepa = dict(zip(bovodepa['especie'].unique(), [2,1,3,4,5,6]))

bovine['grupoesp'] = bovine['especie'].replace(dicine,regex=True)
bovodepa['grupoesp'] = bovodepa['especie'].replace(dicodepa,regex=True)

bovodepanew = get_new_odepa_df(bovine, bovodepa, 'codcom')
bovodepanew['ganado'] = 'bov'
bovodepanew = bovodepanew.join(comunas_shp.set_index('codcom'), on='codcom')
bovts = bovodepanew.pivot_table(columns = 'Ano',
                                values = 'cabezas', index='codreg',
                                aggfunc='sum')

#%% --------------- odepa otros
otrine = df_ine[df_ine['ganado'].isin(['cap', 'ovi', 'otr'])].copy()
otrodepa = unpivot(df_odepa_otr).copy()

dicine = dict(zip(otrine['especie'].unique(), [1,1,1,1,3,
                                               4,4,4,4,4,
                                               5,5,5,5,5,
                                               5,5,2,2,
                                               2,2,2,2,2]))
dicodepa = dict(zip(otrodepa['especie'].unique(), [2,3,4,1]))

otrine['grupoesp'] = otrine['especie'].replace(dicine,regex=True)
otrodepa['grupoesp'] = otrodepa['especie'].replace(dicodepa,regex=True)

otrodepanew = get_new_odepa_df(otrine, otrodepa, 'codcom')
otrodepanew['ganado'] = 'otr'

otrodepanew = otrodepanew.join(comunas_shp.set_index('codcom'), on='codcom')
otrts = otrodepanew.pivot_table(columns = 'Ano',
                                values = 'cabezas', index='codreg',
                                aggfunc='sum')

# #----------- Compute total headcount and group by livestock kind and region
# region_head_totals = df[['codreg', 'ganado' ,'especie', 'cabezas']].groupby(['codreg', 'ganado', 'especie']).sum()
# print('Total de cabezas por especie y por region')
# display(region_head_totals)

# #----------- Compute total headcount per livestock kind per region
# region_stock_totals = df[['codreg', 'ganado', 'cabezas']].groupby(['codreg', 'ganado']).sum()
# print('Total de ganado por region (se agrupan especies)')
# display(region_stock_totals)

# #------------ Compute relative percentage of heads relative to kind of livestock per region
# reg_rel_2_stock = region_head_totals / region_head_totals.groupby(level=[0,1]).transform('sum')
# print('Porcentaje especie relativo al tipo de ganado por region')
# display(reg_rel_2_stock)

# #------------ Compute national total of livestock kind
# nat_stock_totals = df[['ganado', 'cabezas']].groupby('ganado').sum()
# print('Total nacional de ganado')
# display(nat_stock_totals)

#%% Formar dataset consolidado mas demandas

dicdda = {'Broilers': 0.31, 'Gallinas': 0.31, 'Pavos': 0.76, 'Otros': 1,
          'Novillos': 35, 'Vacas': 45, 'Bueyes':45, 'Toros/torunos':45,
          'Vaquillas':27, 'Ovinos':4.5, 'Porcinos': 30, 'Equinos':45,
          'Caprinos': 3.5} # consumo animal en L/cabeza/dia

def calculate_dda(row):
    x = round(row['cabezas'] * dicdda[row['especie']])
    return x


dfodepa = pd.concat([aveodepanew, bovodepanew, otrodepanew], ignore_index=True)
dfodepa.dropna(inplace=True)
dfodepa.drop('Region', axis=1, inplace=True)
dfodepa['codcom'] = dfodepa['codcom'].astype(int)
dfodepa['demanda'] = dfodepa.apply(lambda row: calculate_dda(row), axis=1)
dfodepa.rename({'demanda': 'demandaLdia'}, axis=1, inplace=True)



# ------------------- Filtrar por las comunas dentro del estudio
comcuencafp = 'porcentaje_comuna_subcuenca.csv'
dfcomcuenca = pd.read_csv(comcuencafp, dtype={'COD_DGA': str})

dfodepaAOHIA=dfodepa[dfodepa['codreg'].isin(['05','13','06','07'])].copy()

dfodepa_ts = dfodepaAOHIA.pivot_table(columns = 'Ano',
                                values = 'cabezas', index=['codreg','ganado'],
                                aggfunc='sum')

# ------------------- Crear columna de fechas
dfodepaAOHIA['date'] = dfodepaAOHIA['Ano'].astype(str) + \
    '-' + dfodepaAOHIA['Mes'].astype(str)
dfodepaAOHIA['date'] = pd.to_datetime(dfodepaAOHIA['date'], format='%Y-%m')

# ------------------- Pivotear
pivot = dfodepaAOHIA.pivot_table(index = 'codcom', columns = 'date',
                           values = 'demandaLdia', aggfunc = 'sum')

# ------------------- obtener porcentaje de subcuenca por comuna
pivot = pivot.join(dfcomcuenca.set_index('codcom'), on='codcom')


# ------------------ multiplicar por el porcentaje de cobertura
mcols = list(set(pivot.columns) - set(['COD_DGA', 'perc_area']))
pivot[mcols] = pivot[mcols].multiply(pivot['perc_area'],axis = 'index')

# ------------------ sumar por cuenca
pivot.drop(labels = ['perc_area'], axis = 1, inplace = True)
pivot = pivot.pivot_table(index = 'COD_DGA', aggfunc = 'sum')

# ----------------- ya tenemos el consumo en Ldia por mes por subcuenca
# ----------------- ahora obtenemos el volumen total por mes
from calendar import isleap, monthrange

for col in pivot.columns:
    ndays = monthrange(col.year, col.month)[1]
    pivot.loc[:,col] = pivot.loc[:,col] * ndays
    # esto nos va a dar el total de litros por mes de cada subcuenca

pivot.columns = pd.to_datetime(pivot.columns)
pivot_yr = pivot.resample('Y', axis = 1).sum()

# recapitulando, tenemos:
    # pivot: demanda en volumen, para cada mes (columna) para cada cuenca (fila)
    # pivot_yr: demanda en volumen, para cada anho (columna) para cada cuenca
    
# obtengamos ahora la demanda instantanea por anho
for col in pivot_yr.columns:
    if isleap(col.year):
        pivot_yr.loc[:,col] = pivot_yr.loc[:,col] / 366 / 24 / 60 / 60
    else:
        pivot_yr.loc[:,col] = pivot_yr.loc[:,col] / 365 / 24 / 60 / 60
        

        
# obtengamos ahora la demanda instantanea por mes
for col in pivot.columns:
    ndays = monthrange(col.year, col.month)[1]
    pivot.loc[:,col] = pivot.loc[:,col] / ndays / 24 / 60 / 60
    
# finalmente crucemos el geodataframe para ubicar las subcuencas en el espacio
# y podamos plotear
fdpGIS = os.path.join('..','Etapa 1 y 2', 'GIS', 'Cuencas_DARH')
fpsubcuencas = os.path.join(fdpGIS,
                            'Subcuencas',
                            'SubCuencas_DARH_2015_AOHIA_ZC.geojson')
fpcuencas = os.path.join(fdpGIS, 'Cuencas',
                         'Cuencas_DARH_2015_AOHIA_ZC.geojson')

gdfcuencas = gpd.read_file(fpcuencas)
gdfsubcuencas = gpd.read_file(fpsubcuencas)

df_TS_LPS = pivot_yr.join(gdfsubcuencas[['COD_DGA', 'NOM_CUENCA']].set_index('COD_DGA'), on= 'COD_DGA')
df_TS_LPS = df_TS_LPS.groupby('NOM_CUENCA').sum()
folder_PEC = os.path.join('..', 'Etapa 1 y 2', 'Demanda', 'PEC')
fp = os.path.join(folder_PEC, 'Demanda_PEC_TS_macrocuencas.xlsx')
df_TS_LPS.to_excel(fp)


fields = ['COD_DGA', 'geometry', 'NOM_DGA']
pivot_yr.index = pivot_yr.index.astype(str)
gdf_yr_totals = gdfsubcuencas[fields].set_index('COD_DGA').join(pivot_yr, on = 'COD_DGA')

# creemos la serie de tiempo de demanda por macrocuenca
ts_mensual_scuencas = pivot.join(gdfsubcuencas[['COD_DGA', 'COD_CUENCA']].set_index('COD_DGA'),on='COD_DGA')
ts_mensual_scuencas = ts_mensual_scuencas.groupby('COD_CUENCA').sum().T

#%% ------------------- Figuras
folder_save = os.path.join(folder_pec, '..')
extension = 'jpg'

fp = os.path.join(folder_save, 'Demanda_PEC_L_s_Subsubcuencas' + '.' + extension)
#
fig, ax = plt.subplots(figsize=(8.5,11))
scol = gdf_yr_totals.columns[-2]
gdf_yr_totals.plot(ax=ax, column = scol, scheme = 'percentiles',
                   cmap = 'Blues', legend = True)
gdfcuencas.plot(ax=ax, fc='none', ec='red', lw=2)
ax.get_legend().update({'title': 'Demanda (L/s)'})
ctx.add_basemap(ax=ax, crs='EPSG:32719',
                source=ctx.providers.Esri.WorldTerrain, zoom=9)

ax.set_title('Demanda uso PECUARIO')
ax.set_xlabel('Coordenada UTM Este (m)')
ax.set_ylabel('Coordenada UTM Norte (m)')
plt.show()
plt.savefig(fp, format = extension, bbox_inches = 'tight',
                pad_inches = 0.1)

fp = os.path.join(folder_save, 'Demanda_PEC_L_s_Macrocuencas' + '.' + extension)
fig, ax = plt.subplots(figsize=(8.5,11))
titles = ts_mensual_scuencas.columns.values
ts_mensual_scuencas.plot(y=['1300','0600', '0701', '0703'],ax=ax,
                         subplots=True, layout=(2,2),
                         title=['Río Maipo', 'Río Rapel',
                                'Río Mataquito', 'Río Maule'],
                         sharex = True, xlabel = 'Fecha', sharey = False, 
                         ylabel = 'Demanda ($L/s$)',legend=False)
ax.set_title('Demanda uso PECUARIO')
plt.suptitle('Demanda agua uso PECUARIO')

plt.show()
plt.savefig(fp, format = extension, bbox_inches = 'tight',
                pad_inches = 0.1)


# ------------------- Guardar

savefp = os.path.join(folder_save, 'Dda_PEC_serietiempo_x_subcuenca.csv')

pivot.to_csv(savefp, sep=',')
