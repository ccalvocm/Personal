# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:23:15 2019

@author: cccm
"""


import win32com.client
import re
import time
import os, shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# funciones

def buscar_archivos(path,f_pp,f_t,modelos):   
    
    for r, d, f in os.walk(path):
        for file in f:
            if '_pr_' in str(file) and 'three_points' not in str(file):
                aux = file[0:file.find('_pr_')]
                modelos_clima.append(aux)
                files_pp.append(os.path.join(r, file))
            if '_tmed_' in file:
                files_t.append(os.path.join(r, file))   

# Carpeta de inputs de modelos
  
inputs_folder = r"E:\Atlas Hacienda\Escenarios\CR2xCMIP5_hist-fut\Corregidas_sesgo"
keys = ['Day', 'Month', 'Year']

        
#
files_pp = []
files_t = []
modelos_clima = []

buscar_archivos(inputs_folder,files_pp,files_t,modelos_clima)    
            
index = np.arange(1,971,1)

P_maule = pd.DataFrame(index=index,columns=None)
T_maule = pd.DataFrame(index=index,columns=None)
P_laja = pd.DataFrame(index=index,columns=None)
T_laja = pd.DataFrame(index=index,columns=None)


for j in range(len(files_pp)):
    
    file_pp = pd.read_csv(files_pp[j])
    file_t = pd.read_csv(files_t[j])

    file_pp = file_pp.groupby(["Year","Month"]).sum() 
    file_pp.reset_index(inplace=True)
    
    file_t = file_t.groupby(["Year","Month"]).mean()     
    file_t.reset_index(inplace=True)
   
    pMaule_aux = file_pp[["Year","Month","Armerillo", "Rio_Cipreses_En_Dasague_Laguna_La_Invernada", "Rio_Melado_En_La_Lancha_Dga"]]
    pMaule_aux.columns = [modelos_clima[j],"Mes", "Armerillo", "Cipreses", "Melado"]
    P_maule = pd.concat([P_maule,pMaule_aux], axis=1)

    tMaule_aux = file_t[["Year","Month","Armerillo"]]
    tMaule_aux.columns = [modelos_clima[j],"Mes", "Armerillo"]
    T_maule = pd.concat([T_maule,tMaule_aux], axis=1)

    pLaja_aux = file_pp[["Year","Month","Abanico"]]
    pLaja_aux.columns = [modelos_clima[j],"Mes", "Tucapel"]
    P_laja = pd.concat([P_laja,pLaja_aux], axis=1)

    tLaja_aux = file_t[["Year","Month","Diguillin"]]
    tLaja_aux.columns = [modelos_clima[j],"Mes", "Diguillin"]
    T_laja = pd.concat([T_laja,tLaja_aux], axis=1)

T_maule.to_csv(inputs_folder+'\\T_maule.csv',index = False) 
P_maule.to_csv(inputs_folder+'\\P_maule.csv',index = False) 
T_laja.to_csv(inputs_folder+'\\T_laja.csv',index = False) 
P_laja.to_csv(inputs_folder+'\\P_laja.csv',index = False)



#    pp_yeso = pd.DataFrame(list(zip(fechas_pp,file_pp['El_Yeso_Embalse'])), columns = ['$DateFormat = dd-mm-yy', 'El_Yeso_Embalse'])
#    pp_SanJose = pd.DataFrame(list(zip(fechas_pp,file_pp['San_Jose_De_Maipo_Reten'])), columns = ['$DateFormat = dd-mm-yy', 'San_Jose_De_Maipo_Reten'])
#    pp_CerroCalan = pd.DataFrame(list(zip(fechas_pp,file_pp['Cerro_Calan'])), columns = ['$DateFormat = dd-mm-yy', 'Cerro_Calan'])
#
#    t_yeso = pd.DataFrame(list(zip(fechas_t,file_t['El_Yeso_Embalse'])), columns = ['$DateFormat = dd-mm-yy', 'El_Yeso_Embalse'])
#    t_calan = pd.DataFrame(list(zip(fechas_t,file_t['Cerro_Calan'])), columns = ['$DateFormat = dd-mm-yy', 'Cerro_Calan'])    
#


#if __name__ == "__main__":
#    main()