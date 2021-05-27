# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# importar librerias
import pandas as pd
import re
import os

# Leer multriples archivos

rutas = ["Derechos_Concedidos_V_Region_DESCARGA_25012021.xls",
         "Derechos_Concedidos_VI_Region_DESCARGA_25012021.xls",
         "Derechos_Concedidos_VII_Region_DESCARGA_25012021.xls",
         "Derechos_Concedidos_XIII_Region_DESCARGA_25012021.xls"]

lista_df = []

for ruta in rutas:
    ruta_final = os.path.join("..",
                              "Etapa 1 y 2",
                              "DAA")
    ruta_df = os.path.join(ruta_final, ruta)
    df = pd.read_excel(ruta_df, header = 6, index_col = 0)
    lista_df.append(df)

df = pd.concat(lista_df, axis = 0)

# leer el archivo excel crudo

# df = pd.read_excel('../Etapa 1 y 2/DAA/Derechos_Concedidos_VI_Region.xls', header = 0,
                   # index_col = 0)

# remover espacios y enters en titulos
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# %%

def clean_headers(dataframe):
        
    header = []
    
    for column in dataframe.columns:
        new_col = str(column).replace('\n', ' ')
        new_col = re.sub(' +', ' ', new_col)
        new_col = new_col.lstrip()
        new_col = new_col.rstrip()
        header.append(new_col)
        
    dataframe.columns = header
    dataframe.reset_index(inplace = True, drop = True)
    
    return dataframe

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def clean_spaces(dataframe, columns):
    
    for column in columns:
        dataframe[column] = dataframe[column].str.lstrip()
        dataframe[column] = dataframe[column].str.rstrip()
        
    return dataframe


def clean_character(dataframe, columns, char1, char2):
    
    for column in columns:
        dataframe[column] = dataframe[column].str.replace(char1, char2)
        
    return dataframe

def column_to_numeric(dataframe, columns):
    
    for column in columns:
        dataframe[column] = pd.to_numeric(dataframe[column])
        
    return dataframe

# %%
df = clean_headers(df)
df.reset_index(drop = True, inplace = True)


columns = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio',
           'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

columns2 = ['Código de Expediente', 'Región', 'Provincia', 'Comuna',
            'Nombre Solicitante', 'Unidad de Resolución/ Oficio/ C.B.R.',
            'Artículo Transitorio', 'Tipo Derecho', 'Naturaleza del Agua',
            'Clasificación Fuente', 'Uso del Agua', 'Cuenca',  'SubCuenca', 
            'SubSubCuenca', 'Fuente', 'Unidad de Caudal', '¿Caudal Promedio Anual?', 
            'Caudal Ecológico (l/s)', '¿Caudal Ecológico Promedio?', 
            'C.B.R.', 'Fojas']


df = clean_spaces(df, columns)
df = clean_character(df, columns, ',', '.')
df = clean_spaces(df, columns2)
df = column_to_numeric(df, columns)
df = clean_character(df, ['Longitud Captación', 'Latitud Captación',
                          'Longitud Restitución', 'Latitud Restitución'],
                     ' ', '')
# df.drop(columns = ['FECHA GRINGA'], inplace = True)
df["Fecha de Resolución/ Envío al Juez/ Inscripción C.B.R."]=pd.to_datetime(df["Fecha de Resolución/ Envío al Juez/ Inscripción C.B.R."],
               format="%d/%m/%Y")
#%%

# Filtro de fechas
fechacorte = "2017/08/03"
prefecha = df["Fecha de Resolución/ Envío al Juez/ Inscripción C.B.R."] < fechacorte
postfecha = ~prefecha


# Filtrar con coordenadas
wCaptacion = ~((df['UTM Este Captación (m)'] == 0) | (df['UTM Norte Captación (m)'] == 0))
wLLCaptacion = ~((df['Longitud Captación'] == '') | (df['Latitud Captación'] == ''))
               
wRestitucion = ~((df['UTM Este Restitución (m)'] == 0) | (df['UTM Norte Restitución (m)'] == 0))
wLLRestitucion = ~((df['Longitud Restitución'] == '') | (df['Latitud Restitución'] == ''))                 


# guardar archivo excel tratado

ruta = os.path.join("..", "Etapa 1 y 2", "DAA")
nombre_archivo = "DAA_filtro_fecha.xlsx"
ruta_final = os.path.join(ruta, nombre_archivo)

with pd.ExcelWriter(ruta_final) as writer:
    df[(wCaptacion | wRestitucion | wLLCaptacion | wLLRestitucion) & prefecha].to_excel(writer, sheet_name = 'c-coord-pre', index = False)
    df[(wCaptacion | wRestitucion | wLLCaptacion | wLLRestitucion) & postfecha].to_excel(writer, sheet_name = 'c-coord-post', index = False)
    df[~(wCaptacion | wRestitucion | wLLCaptacion | wLLRestitucion) & prefecha].to_excel(writer, sheet_name = 's-coord-pre', index = False)
    df[~(wCaptacion | wRestitucion | wLLCaptacion | wLLRestitucion) & postfecha].to_excel(writer, sheet_name = 's-coord-post', index = False)


