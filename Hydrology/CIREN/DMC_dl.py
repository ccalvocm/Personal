# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:25:23 2020

@author: ccalvo
"""

import requests
import zipfile
import os
import random
from time import sleep
import pandas as pd
import random
import numpy as np

def bajarDMCAutomaticas():    
    
    estaciones = ['330010','330011','330012','330013','330015','330016','330017','330018','330019','330020','330021',
                  '330022','330023','330024','330025','330026','330027','330029','330032','330033','330034','330035',
                  '330036','330037','330038','330039','330040','330041','330046','330047','330050','330051','330052',
                  '330054','330056','330057','330058','330060','330061','330063','330064','330065','330068','330069',
                  '330070','330071','330072','330075','330076','330077','330078','330081','330082','330083','330084',
                  '330085','330093','330095','330110','330111','330112','330113','330114','330115','330116','330117',
                  '330118','330119','330121','330122','330130','330131','330132','330133','330134','330135','330136',
                  '330137','330138','330140','330141','330142','330143','330144','330145','330146','330147','330148',
                  '330149','330150','330151','330152','330153','330154','330155','330156','330157','330158','330159',
                  '330160','330162','330163','330164','330165','330166','330167','330168','330169','330170','330171',
                  '330172','330173','330174','330175','330176','330177','330178','330179','330180','330181','330182',
                  '330183','330184','330190','330901','330903','330905','330913','330964','340046','340146','330042',
                  '330192','340001','340002','340003','340004','340005','340006','340007','340008','340009','340010',
                  '340011','340012','340013','340014','340015','340016','340017','340018','340019','340020','340021',
                  '340022','340024','340025','340032','340033','340034','340035','340036','340037','340038','340039',
                  '340040','340041','340042','340044','340045','340047','340048','340049','340050','340051','340052',
                  '340053','340055','340056','340057','340058','340059','340061','340062','340063','340064','340065',
                  '340066','340067','340068','340069','340070','340071','340072','340073','340074','340075','340077',
                  '340093','340094','340095','340096','340097','340098','340099','340100','340101','340102','340103',
                  '340104','340105','340106','340108','340109','340113','340114','340115','340116','340117','340118',
                  '340119','340120','340121','340122','340123','340124','340125','340126','340127','340128','340129',
                  '340130','340131','340132','340133','340134','340135','340136','340137','340147','340148','340149',
                  '340902','340904','340905','340906','340922','340023','340026','340027','340028','340029','340030',
                  '340031','340107','340110','340111','340112','340138','340139','340141','340142','340143','340144',
                  '340145','350001','350002','350003','350004','350005','350006','350007','350008','350009','350010',
                  '350011','350012','350014','350015','350016','350017','350018','350020','350021','350023','350024',
                  '350025','350026','350027','350028','350029','350030','350031','350032','350033','350034','350035',
                  '350036','350037','350038','350039','350040','350041','350050','350051','350052','350054','350055',
                  '350056','350057','350058','350059','350060','350061','350062','350063','350064','350065','350066',
                  '350067','350068','350069','350070','350071','350072','350073','350074','350075','350076','350077',
                  '350078','350079','350080','350081','350082','350083','350084','350085','350086','350087','350088',
                  '350089','350090','350091','350092','350093','350094','350095','350096','350097','350902','350906',
                  '350909','360001','360002','360003','360004','360005','360006','360007','360033','360043','360047',
                  '360048','360049','360050','360051','360052','360053','360054','360055','360056','360057','360058',
                  '360079','360080','320041','320051','330007','330030','330031','330066','360011','360019','360042']
    
    ruta_GitHub = 'D:\GitHub'
    ruta_GitHub = 'C:\Users\ccalvo\Documents\GitHub'
    ruta_descargas = ruta_GitHub+r'\Analisis-Oferta-Hidrica\Otros\Downloads'
    os.chdir(ruta_descargas)
    
    for estacion in estaciones:
        sleep(random.randint(10,20)) #NO CAMBIAR
    
        URLdatos = 'https://climatologia.meteochile.gob.cl/application/productos/gethistoricos/'+estacion+r'_XXXX_Agua24Horas_'
        r = requests.get(URLdatos, stream = True)
        textoURL = r.text
        
        if not 'no encontrada' in textoURL:
            with open(ruta_descargas+"\\Pp"+estacion+".zip","wb") as zipf:
                 for chunk in r.iter_content():
                     zipf.write(chunk)
            zipf.close()
    
    
    for file in os.listdir(ruta_descargas):
        zip_ref = zipfile.ZipFile(os.path.abspath(file)) # create zipfile object
        zip_ref.extractall(ruta_descargas) # extract file to dir
        zip_ref.close() # close file
        os.remove(file) # delete zipped file

def bajarDMCOtras():

    ruta = r'C:\Users\ccalvo\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos\datosDMC\V'
       
    estaciones = ['330010','330011','330012','330013','330015','330016','330017','330018','330019','330020','330021',
                  '330022','330023','330024','330025','330026','330027','330029','330032','330033','330034','330035',
                  '330036','330037','330038','330039','330040','330041','330046','330047','330050','330051','330052',
                  '330054','330056','330057','330058','330060','330061','330063','330064','330065','330068','330069',
                  '330070','330071','330072','330075','330076','330077','330078','330081','330082','330083','330084',
                  '330085','330093','330095','330110','330111','330112','330113','330114','330115','330116','330117',
                  '330118','330119','330121','330122','330130','330131','330132','330133','330134','330135','330136',
                  '330137','330138','330140','330141','330142','330143','330144','330145','330146','330147','330148',
                  '330149','330150','330151','330152','330153','330154','330155','330156','330157','330158','330159',
                  '330160','330162','330163','330164','330165','330166','330167','330168','330169','330170','330171',
                  '330172','330173','330174','330175','330176','330177','330178','330179','330180','330181','330182',
                  '330183','330184','330190','330901','330903','330905','330913','330964','340046','340146','330042',
                  '330192','340001','340002','340003','340004','340005','340006','340007','340008','340009','340010',
                  '340011','340012','340013','340014','340015','340016','340017','340018','340019','340020','340021',
                  '340022','340024','340025','340032','340033','340034','340035','340036','340037','340038','340039',
                  '340040','340041','340042','340044','340045','340047','340048','340049','340050','340051','340052',
                  '340053','340055','340056','340057','340058','340059','340061','340062','340063','340064','340065',
                  '340066','340067','340068','340069','340070','340071','340072','340073','340074','340075','340077',
                  '340093','340094','340095','340096','340097','340098','340099','340100','340101','340102','340103',
                  '340104','340105','340106','340108','340109','340113','340114','340115','340116','340117','340118',
                  '340119','340120','340121','340122','340123','340124','340125','340126','340127','340128','340129',
                  '340130','340131','340132','340133','340134','340135','340136','340137','340147','340148','340149',
                  '340902','340904','340905','340906','340922','340023','340026','340027','340028','340029','340030',
                  '340031','340107','340110','340111','340112','340138','340139','340141','340142','340143','340144',
                  '340145','350001','350002','350003','350004','350005','350006','350007','350008','350009','350010',
                  '350011','350012','350014','350015','350016','350017','350018','350020','350021','350023','350024',
                  '350025','350026','350027','350028','350029','350030','350031','350032','350033','350034','350035',
                  '350036','350037','350038','350039','350040','350041','350050','350051','350052','350054','350055',
                  '350056','350057','350058','350059','350060','350061','350062','350063','350064','350065','350066',
                  '350067','350068','350069','350070','350071','350072','350073','350074','350075','350076','350077',
                  '350078','350079','350080','350081','350082','350083','350084','350085','350086','350087','350088',
                  '350089','350090','350091','350092','350093','350094','350095','350096','350097','350902','350906',
                  '350909','360001','360002','360003','360004','360005','360006','360007','360033','360043','360047',
                  '360048','360049','360050','360051','360052','360053','360054','360055','360056','360057','360058',
                  '360079','360080','320041','320051','330007','330030','330031','330066','360011','360019','360042']

# ===========================
# Variables que deseen bajar
# ===========================


#    variable = 'Agua Caida, Total Diario'
#    variable = 'Temperatura del Aire Seco'
#    variable = 'Temperatura Mínima AM'
#    variable = '2da. TempMín (Temp. Mínima PM)'
#    variable = '2da. TempMáx (Temp. Máxima AM)'
#    variable = 'Temperatura Máxima PM'
    variable = 'Viento a 10 m. de Altura'
               
    producto = {
                'Agua Caida, Total Diario': ['60','125'], #precipitación cada 24 horas
                'Temperatura del Aire Seco' : ['26', '58'],

                'Temperatura Mínima AM' : ['42', '97'],
                '2da. TempMín (Temp. Mínima PM)' : ['43', '98'],
                '2da. TempMáx (Temp. Máxima AM)' : ['44', '100'],
                'Temperatura Máxima PM' : ['45', '102'],
                'Viento a 10 m. de Altura' : ['28','61']
                } 

    url = 'https://climatologia.meteochile.gob.cl/application/informacion/datosMensualesDelElemento'
    user_agent_list = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:82.0) Gecko/20100101 Firefox/82.0'
    ]      

    for estacion in estaciones:
        all_data = pd.DataFrame([])             
        
        direccion_ficha_est = r'https://climatologia.meteochile.gob.cl/application/informacion/ficha-de-estacion'+'/'+estacion
        sleep(random.randint(5,10)) #NO CAMBIAR
        User_agent =  random.choice(user_agent_list)

        r = requests.get(direccion_ficha_est, headers = {'User-Agent': User_agent}, stream = True)
        
        nombre = pd.read_html(r.text)[1]
        coordenadas = pd.read_html(r.text)[2]
        ficha = pd.concat([nombre, coordenadas])
        
        desde_hasta = pd.read_html(r.text)[6]
        indice = desde_hasta['Nombre'][desde_hasta['Nombre'] == variable].dropna().index
        
        if (len(indice) > 0) and (desde_hasta['Información Disponible','Desde'].loc[indice].values[0] != '.'):
            year_i = int(desde_hasta['Información Disponible','Desde'].loc[indice])
            year_f = int(desde_hasta['Información Disponible','Hasta'].loc[indice])
            agnos = np.arange(year_i, year_f+1)
            agnos = agnos[agnos > 1978]
            
            if len(agnos) > 0:
                for agno in agnos:
                    print(agno)
                    for mes in np.arange(1,13,1):
                        direcccion = url+'/'+estacion+'/'+str(agno)+'/'+str(mes)+'/'+producto[variable][0]
                        sleep(random.randint(3,5)) #NO CAMBIAR
                        User_agent =  random.choice(user_agent_list)
    
                        r = requests.get(direcccion, headers = {'User-Agent': User_agent}, stream = True)
                        textoURL = pd.read_html(r.text)
                        all_data = all_data.append(textoURL,ignore_index=True)
                
                if (variable == 'Temperatura del Aire Seco') | (variable == 'Viento a 10 m. de Altura'):
                    all_data.index =  pd.to_datetime(all_data['Fecha']+' '+all_data['Hora (UTC)'], dayfirst = True)
                else:
                    all_data.index =  pd.to_datetime(all_data['Fecha'], dayfirst = True)
                    
                all_data.sort_index(inplace=True)
                all_data = all_data.loc[all_data.index.notnull()]
                
                writer = pd.ExcelWriter(ruta+'\\'+variable+'_estacion_'+estacion+'.xlsx', engine='xlsxwriter')
                
                # Write each dataframe to a different worksheet.
                ficha.to_excel(writer, sheet_name='ficha_est')
                all_data.to_excel(writer, sheet_name='Datos')
                
                # Close the Pandas Excel writer and output the Excel file.
                writer.save()

def ordenarMOP(ruta = r'C:\Users\ccalvo\OneDrive - ciren.cl\HM\RM\Temperatura\Rio Volcan en Queltehues', file = 'T_rio_Volcan_en_Queltehues.csv'):
        
    os.chdir(ruta)
    all_data = pd.DataFrame([])
    
    for filename in os.listdir(ruta):
        archivo = pd.read_html(ruta+'\\'+filename)[0]
        archivo.index = pd.to_datetime(archivo['Fecha'])
        all_data = all_data.append(archivo,ignore_index=True)
    all_data.index =  pd.to_datetime(all_data['Fecha-Hora de Medicion'], dayfirst = True)
    all_data.sort_index(inplace=True)
    all_data.iloc[:,2:] = all_data.iloc[:,2:]/1000.
    all_data.to_csv(file)

def ordenarDMC(ruta, file):
    
    # ==================
    # Precipitaciones
    # ==================
    
    ruta = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos\datosDMC\pp'
    
    all_data = pd.DataFrame([], index = pd.date_range('1900-01-01','2021-04-01', freq = '1d'))   
    meta_data = pd.DataFrame([], index = ['Codigo','Pseudonimo','Lat','Lon','Alt','Zone'])
    
    for filename in os.listdir(ruta):
        if (filename.endswith(".xlsx")) & ('pp_DMC_daily' not in filename): 
            estacion = filename[-11:-5]
            df = pd.read_excel(ruta+'\\'+filename, sheet_name = 'Datos', parse_dates = True, index_col = 0)
            metadata = pd.read_excel(ruta+'\\'+filename, sheet_name = 'ficha_est', parse_dates = True, index_col = 0)
            metadata.iloc[8,1] = int(metadata.iloc[8,1].strip(' Mts.'))
            meta_data[estacion] = metadata.iloc[[0,3,6,7,8,11],1].values
            all_data.loc[df.index,estacion] = df['RRR24']
   
    all_data = all_data.replace('.',np.nan)
    all_data = all_data.loc[all_data.index > '1979-01-01']
    meta_data = meta_data.transpose()
    
    
    # ===============================
    #           Por cuenca
    # ===============================
    
    cuenca = 'Maule'
    
    est_cuenca = pd.read_csv(ruta+r'\\DMC_'+cuenca+'.csv')['Codigo'].astype(str)
    
    meta_cuenca = meta_data[meta_data['Codigo'].isin(est_cuenca)]
    data_cuenca = all_data[meta_cuenca['Codigo']]
    
    # ------Guardar
    
    writer = pd.ExcelWriter(ruta+'\\pp_DMC_daily_'+cuenca+'.xlsx', engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    meta_cuenca.to_excel(writer, sheet_name='ficha_est')
    data_cuenca.to_excel(writer, sheet_name='Datos')
    
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    writer.close()

def ordenarCLIMATOL(ruta):
    cuenca = 'Maipo'
    
    ruta = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Clima\Pp\Consolidado'+'\\'+cuenca
    pp_consolidado = pd.read_excel(ruta+'\\'+cuenca+'_consolidado.xlsx',index_col = 0, sheet_name = 'data')
    pp_consolidado = pp_consolidado.loc[pp_consolidado.index < '2020-07-01']
    pp_consolidado = pp_consolidado.loc[pp_consolidado.index > '1989-12-31']

    pp_consolidado.to_csv(ruta+'\\'+cuenca+'_consolidado.dat', sep = ' ', index = False, header = None)
    
    metadata = pd.read_excel(ruta+'\\'+cuenca+'_consolidado.xlsx', sheet_name = 'info estacion')
    meta = metadata[['Longitud','Latitud','Altitud','Codigo Estacion','Nombre estacion']]
    
    for i, row in meta.iterrows():
        
        nombre = meta.loc[i,'Codigo Estacion']
        
        try:
            nombre = meta.loc[i,'Codigo Estacion'].split("-")[0]
        except:
            aaaa = 1
            
        meta.loc[i,'Codigo Estacion'] = '\"'+str(nombre)+'\"'
        meta.loc[i,'Nombre estacion'] = '\"'+str(meta.loc[i,'Nombre estacion'])+'\"'
    
    meta.replace('\"\"\"','\"')
    meta.to_csv(ruta+'\\'+cuenca+'_consolidado.est', sep = ' ', index = False, header = None)
