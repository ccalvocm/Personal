# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 23:25:47 2021

@author: Carlos
"""

#########################
###     Preámbulo     ###
#########################

import scipy
import pandas as pd
import numpy as np
import os
import lxml
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import geopandas

#########################
## Graficar con comas ###
#########################
#Locale settings
import locale
# Set to Spanish locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "es_ES")

plt.rcdefaults()

# Tell matplotlib to use the locale we set above
plt.rcParams['axes.formatter.use_locale'] = True


def identificarErrores(df):
    mask_errores = df[df.iloc[:,-1].str.contains("\*").replace(np.nan,False)].index
    df.iloc[:,-2].loc[mask_errores] = np.nan #columna primero con iloc
    return df

def copyIndex(df):
    for col in df:
        df.loc[df.index,col] = df.index.daysinmonth
    return df
    
def pronostico(df,meses, orden):    # AR example
    # SARIMA 
    # fit model
    model = SARIMAX(df, order = orden, seasonal_order=(1, 1, 3, 12))
    model_fit = model.fit(disp=False)
    # make prediction
    return model_fit.forecast(meses)

def pronostico_ARMA(df,meses, orden):
    # fit model
    model = ARIMA(df, order=orden)
    model_fit = model.fit()
    # make prediction
    return model_fit.forecast(meses), model_fit.get_forecast(meses).conf_int(alpha=0.3)

def flags_mon(df):
    df_flag = df.copy()
    df_flag[:] = 1
    df_flag[df.isnull()] = 0
    df_flag = df_flag.resample('MS').sum()
    df_mon = df.copy().apply(pd.to_numeric).resample('MS').mean()[df_flag > 20]
    return df_mon

# ====================================================
def completarVIC(ruta_VIC, df, estacion):
# ----------------------------------------------------
# completar con VIC
# ruta_VIC : la ruta de los q de VIC
# df : dataframe de la estación a complementar
# estacion : estacion a complementar
# ----------------------------------------------------

    q_VIC = pd.read_csv(ruta_VIC, parse_dates = True, index_col = 0)
    
    df_2 = pd.DataFrame([], index = pd.date_range('1979-01-01',max(max(df.index),pd.to_datetime('2015-12-01')),freq = 'MS'), columns = [estacion])
    df_2.loc[df.index,estacion] = df
    
    idx = df_2[estacion][df_2[estacion].isnull()].index.intersection(q_VIC.index)
    df_2.loc[idx,estacion] = q_VIC.loc[idx,'Salida_'+estacion[1:-2]]
    
    return df_2
    
def main():
    
    '''
    #======================================
    #               Preámbulo
    #======================================
    '''
    
    dicc_DAA = {'cmrEnero' : 1,	'cmrFebrero' : 2, 'cmrMarzo' : 3, 'cmrAbril' : 4, 'cmrMayo' : 5, 'cmrJunio' : 6, 'cmrJulio' : 7, 
                'cmrAgosto' : 8,	'cmrSeptiem' : 9, 	'cmrOctubre' :10, 	'cmrNoviemb' : 11,	'cmrDiciemb' : 12}
    
    #-----------------Rutas
    ruta_OD = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2'
    #ruta_OD = r'C:\Users\ccalvo\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2'
    os.chdir(ruta_OD+r'\datos')
    ruta_DAA_Olivares_Colorado = ruta_OD + r'\datos\Bocatomas_Olivares_Colorado.xlsx'
    
    ruta_QCF = ruta_OD+r'\datos\Datos CF\Datos_BNA_EstacionesDGA\BNAT_CaudalDiario.txt'
    ruta_QCF = ruta_OD+r'\datos\formatoDGA.txt'
    
    # ----------------outputs
    ruta_out = ruta_OD.replace('Etapa 1 y 2','') + r'scripts\outputs\caudales'
    
    #-----------------Maipo
    ruta_Maipo_day = ruta_OD + r'\datos\datosDGA\Q\Maipo\Maipo_cr2corregido_Q.xlsx'
    ruta_DAA_Maipo = ruta_OD + r'\datos\DAA_Maipo_RVQ_ROJC_RCRO_RMLM_RMLH.csv'
       
    #-----------------Rapel
    ruta_CS = ruta_OD + r'\datos\RIO RAPEL_Q_diario.csv'
    ruta_Rapel = ruta_OD + r'\datos\Q_relleno_MLR_Rapel_1979-2020_outlier_in_correction_median.csv'
    ruta_RCPT_NAT_DGA = ruta_OD + r'\datos\R_CachapoalenPuenteTermas_RN.xlsx'
    ruta_Rapel_mon = ruta_OD + r'\datos\RIO RAPEL_Q_mensual.xlsx'
    ruta_Rapel_VIC_mon = ruta_OD + r'\datos\q_Rapel_VIC_mon.csv'
    ruta_Chimbarongo_VIC = ruta_OD + r'\datos\VIC\Estero Chimbarongo\Estero_Chimbarongo_VIC.csv'
    
    #-----------------Mataquito
    ruta_Mataquito_daily = ruta_OD + r'\datos\RIO MATAQUITO_diario.csv'
    ruta_Planchon = ruta_OD + r'\datos\El_Planchon_JVRT.csv'
    ruta_Planchon_CNR = ruta_OD + r'\datos\Entrega_El_Planchon.xlsx'
    ruta_Mataquito_VIC_mon = ruta_OD + r'\datos\q_Mataquito_VIC_mon.csv'
    
    #-----------------Maule
    ruta_Maule_relleno = ruta_OD+r'\datos\Q_relleno_MLR_Maule_1979-2020_outlier_in_correction_median.csv'
    ruta_Maule_crudo = ruta_OD+r'\datos\Q_Maule_1900-2020.xlsx'
    ruta_Maule_original = ruta_OD+r'\datos\RIO MAULE_diario.csv'
    ruta_MauleArmerillo_rec = ruta_OD+r'\Solicitudes de información\Recibidos\JDV Río Maule\Estadistica Maule 09-17.xlsx'
    ruta_MauleArmerillo_NAT_DGA = ruta_OD+r'\datos\QMA_NAT_DGA.csv'
    ruta_Maule_mon = ruta_OD + r'\datos\RIO MAULE_mensual.xlsx'
    ruta_Vols = ruta_OD+r'\datos\Volumenes_embalses_Maule'
    ruta_MA_RN = ruta_OD+r'\datos\Q_MA_RN_mensual.txt'
    ruta_Maule_VIC_mon = ruta_OD + r'\datos\Mauleq_Maule_VIC_mon.csv'
    
    #-----------------Ñuble
    
    ruta_q_Nuble = ruta_OD+r'\datos\RIO ÑUBLE_Q_mensual.xlsx'
    ruta_DAA_Nuble = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\SIG\Ñuble\DAA_ÑUBLE.shp'

    #-----------------Generación
    ruta_gen = ruta_OD+r'\Demanda\ELE\Fuentes informacion\Generacion_Bruta.xlsx'
    ruta_melado = ruta_OD+r'\datos\Datos CF\Datos_BNA_EstacionesDGA\BNAT_Embalses.txt'
    
    # ----------------DAA
    ruta_DAA = ruta_OD + r'\DAA\DERECHOS_DE_APROVECHAMIENTO_DE_AGUAS_19052021\DAA_Maipo.shp'
    
    # ----------------subcuencas
    ruta_basins = ruta_OD.replace('Etapa 1 y 2','') + r'SIG\Cuencas_CAMELS\Cuencas_cabecera_MaipoRapelMataquitoMaule.shp'
    
  
    '''
    # =========================================
    # Naturalización de la cuenca del río Rapel 
    # ==========================================
    '''
    # río Cachapoal en Pte Termas 06008009-7
    # ---------------------------------------------
    
    q_Rapel_mon = pd.read_excel(ruta_Rapel_mon, parse_dates = True, index_col = 0, sheet_name  = 'data')
    q_Rapel_mon = q_Rapel_mon[q_Rapel_mon.columns[2:]]
    q_Rapel_mon_flags = pd.read_excel(ruta_Rapel_mon, parse_dates = True, index_col = 0, sheet_name  = 'info data')
    q_Rapel_mon[q_Rapel_mon_flags < 20] = np.nan
    q_RCHLN_DGA = q_Rapel_mon['06013001-9'].copy()

        
    q_RCPT_mon = q_Rapel_mon['06008005-4'].copy()
    
    # caudal aduccion Sauzal en Pte Termas
    q_CS = pd.read_csv(ruta_CS, parse_dates = True, index_col = 0)['06008009-7'].resample('MS').mean()
    q_CS = q_CS[q_CS.notna()]
    q_CS = q_CS.loc[q_CS.index <= max(q_RCPT_mon.index)]
    
    # caudal central Sauzal
    caudales_gen = pd.read_csv(r'.\caudales_generacion_hidro.csv', index_col = 0, parse_dates = True)
    caudales_gen_sauzal = caudales_gen['SAUZAL']
    
    # sumar caudal aduccion Sauzal a Río Cachapoal en Pte Termas
    q_RCPT_mon.loc[q_RCPT_mon.index < min(caudales_gen_sauzal.index)] = np.nan
    q_RCPT_NAT_DGA = pd.read_excel(ruta_RCPT_NAT_DGA, parse_dates = True, index_col = 0)
    q_RCPT_NAT_DGA = q_RCPT_NAT_DGA.loc[(q_RCPT_NAT_DGA.index >= min(q_RCPT_mon.index)) & (q_RCPT_NAT_DGA.index <= max(q_RCPT_mon.index))]
    q_RCPT_nat = q_RCPT_mon.copy()
    q_RCPT_nat.loc[q_RCPT_nat.index] = np.nan
    q_RCPT_nat.loc[q_RCPT_NAT_DGA.index] = q_RCPT_NAT_DGA['QMANAT']
    
    # sumar caudal Sauzal CEN
    idx = caudales_gen_sauzal.index.intersection(q_RCPT_mon.index)
    q_RCPT_nat.loc[idx] = q_RCPT_mon.copy().loc[idx] + caudales_gen_sauzal.copy().loc[idx] #suma canal Sauzal
    
    # sumar caudal Sauzal DGA
    idx =  q_CS.index.intersection(q_RCPT_mon.index)
    q_RCPT_nat.loc[idx] = q_RCPT_mon.copy().loc[idx] + q_CS.copy().loc[idx] #suma canal Sauzal
        
    idx = pd.to_datetime(['2002-11-01', '2002-12-01' ,'2003-01-01'])
    q_RCPT_nat.loc[idx] = q_Rapel_mon.loc[idx,'06008005-4']
    
    q_Rapel_mon['06008005-4'] = np.nan
    q_Rapel_mon.loc[q_RCPT_nat[q_RCPT_nat.notnull()].index,'06008005-4'] = q_RCPT_nat[q_RCPT_nat.notnull()]

    # ================================================================
    #   Naturalización Río Claro en Hacienda Las Nieves 06013001-9
    # ================================================================
    
    q_VIC = pd.read_csv(ruta_Rapel_VIC_mon, parse_dates = True, index_col = 0)
    q_RCHLN_mon_VIC = q_VIC['Salida_6013001']
    
    q_Rapel_mon['06013001-9'] = np.nan
    q_Rapel_mon['06013001-9'] = q_RCHLN_mon_VIC
    idx = pd.to_datetime(['2008-06-01','2008-07-01','2008-08-01'])
    q_Rapel_mon.loc[idx, '06013001-9'] = q_RCHLN_DGA.loc[idx]
    
    
    # ================================================================
    #   Naturalización Estero Chimbarongo 06034022-6
    # ----------------------------------------------------------------
    # Se suma el caudal observado en Estero Chimbarongo con los 
    # afluentes del Estero Chimbarongo y se le resta el 
    # trasvase del canal Teno
    # pero se necesita la regulación del embalse => VIC
    # ----------------------------------------------------------------
    
    # #--------Trasvase Teno Chimbarongo
 
    # q_CTK13 = flags_mon(pd.read_csv(ruta_CS, parse_dates = True, index_col = 0)['06033009-3'])
    # idx = q_CTK13[q_CTK13.notnull()].index
      
    # idx2 = list(set(idx.intersection(q_Rapel_mon['06033011-5'].notnull().index).intersection(q_Rapel_mon['06034022-6'].notnull().index)))
    
    # q_Rapel_mon_copy = q_Rapel_mon.copy()
    q_Rapel_mon['06034022-6'] = np.nan
    chimbarongoVIC = pd.read_csv(ruta_Chimbarongo_VIC, index_col = 0, parse_dates = True)
    idx2 = chimbarongoVIC[chimbarongoVIC.notnull()].index
    q_Rapel_mon.loc[idx2,'06034022-6'] = chimbarongoVIC.iloc[:,0]
    
    # q_Rapel_mon.loc[idx2,'06034022-6'] = q_Rapel_mon_copy.loc[idx2,'06034022-6']-q_CTK13.loc[idx2]+q_Rapel_mon_copy.loc[idx2,'06033011-5']   
    # q_Rapel_mon['06034022-6'][q_Rapel_mon['06034022-6'] < 0] = 0
        
    
    del q_Rapel_mon['06033009-3']
            
    # ===========================================================
    #               sumar derechos de riego 
    # ===========================================================

    DAA_OH = geopandas.read_file(ruta_DAA_Maule)
    DAA_OH = DAA_OH[DAA_OH['Region'] == 6]
    DAA_OH = DAA_OH[(DAA_OH['Tipo_Derec'] == 'Consuntivo') & (DAA_OH['Ejercici_2'] == 'Continuo')]
              
    for ind,col in q_RCPT_nat.items():
        DAA_fecha = DAA_OH.copy()[pd.to_datetime(DAA_OH['Fecha_de_R']) < ind]

        DAA_fecha_RCPT  = DAA_fecha.copy()[DAA_fecha['Estacion'].isin(['06006001-0','06000001-8','06000003-4',
        '06002001-9', '06003001-4', '06008009-7', '06008005-4'])]
        DAA_fecha_RTBLB = DAA_fecha.copy()[DAA_fecha['Estacion'].isin(['06028001-0'])]
        DAA_fecha_RCEV = DAA_fecha.copy()[DAA_fecha['Estacion'].isin(['06027001-5'])]
        DAA_fecha_PEP = DAA_fecha.copy()[DAA_fecha['Estacion'].isin(['06006001-0'])]
        DAA_fecha_RLLAJC  = DAA_fecha.copy()[DAA_fecha['Estacion'].isin(['06000003-4','06000001-8'])]
        DAA_fecha_RCAJC = DAA_fecha.copy()[DAA_fecha['Estacion'].isin(['06002001-9'])]
        DAA_fecha_RC5JC  = DAA_fecha.copy()[DAA_fecha['Estacion'].isin(['06003001-4','06000003-4','06002001-9','06000001-8'])]        
        DAA_fecha_RCT = DAA_fecha.copy()[DAA_fecha['Estacion'].isin(['06015001-K','06013001-9'])]     
        DAA_fecha_EZPEN = DAA_fecha.copy()[DAA_fecha['Estacion'].isin(['06018001-6'])]     
                
        # En lugar del mes se debe usar el derecho promedio, por las pifias del CPA DGA
        mes_DAA = 'Caudal__l_'
        mes_DAA = list(dicc_DAA.keys())[list(dicc_DAA.values())[ind.month-1]-1]
        
        q_derechos_RCPT = DAA_fecha_RCPT[mes_DAA].dropna().sum()/1000.
        q_derechos_RTBLB = DAA_fecha_RTBLB[mes_DAA].dropna().sum()/1000.
        q_derechos_RCEV = DAA_fecha_RCEV[mes_DAA].dropna().sum()/1000.
        q_derechos_PEP = DAA_fecha_PEP[mes_DAA].dropna().sum()/1000.
        q_derechos_RLLAJC = DAA_fecha_RLLAJC[mes_DAA].dropna().sum()/1000.
        q_derechos_RCAJC = DAA_fecha_RCAJC[mes_DAA].dropna().sum()/1000.
        q_derechos_RC5JC = DAA_fecha_RC5JC[mes_DAA].dropna().sum()/1000.
        q_derechos_RCT = DAA_fecha_RCT[mes_DAA].dropna().sum()/1000.
        q_derechos_EZPEN = DAA_fecha_EZPEN[mes_DAA].dropna().sum()/1000.


        print(q_derechos_RCPT,q_derechos_RTBLB,q_derechos_RCEV)

        q_Rapel_mon.loc[ind,'06008005-4'] += q_derechos_RCPT
        q_Rapel_mon.loc[ind,'06028001-0'] += q_derechos_RTBLB
        q_Rapel_mon.loc[ind,'06027001-5'] += q_derechos_RCEV
        q_Rapel_mon.loc[ind,'06006001-0'] += q_derechos_PEP
        q_Rapel_mon.loc[ind,'06000003-4'] += q_derechos_RLLAJC
        q_Rapel_mon.loc[ind,'06002001-9'] += q_derechos_RCAJC
        q_Rapel_mon.loc[ind,'06003001-4'] += q_derechos_RC5JC
        q_Rapel_mon.loc[ind,'06015001-K'] += q_derechos_RCT #Río Claro en Tunca
        q_Rapel_mon.loc[ind,'06018001-6'] += q_derechos_EZPEN # ESTERO ZAMORANO EN PUENTE EL NICHE
    
    # ----------Reemplazar con VIC
    
    estaciones = ['06033011-5','06043001-2', '06019003-8','06011001-8'] 

    for est in estaciones:
        q_Rapel_mon[est] = np.nan
        idx = q_VIC.index
        q_Rapel_mon.loc[idx,est] = q_VIC.loc[idx,'Salida_'+est[1:-2]]
    
    # el 1997 del VIC da mal
    q_Rapel_mon.loc[q_Rapel_mon.index.year == 1997, estaciones] = np.nan
        
    # ----------completar con VIC
    estaciones2 = ['06015001-K','06018001-6','06006001-0','06008005-4'] #'','','Pangal en Pangal','Rio cachapoal en Puente Termas'
    for est in estaciones2:
        q_vic = completarVIC(ruta_Rapel_VIC_mon, q_Rapel_mon[est], est)
        if est == '06006001-0':
            q_vic = q_vic.loc[q_vic.index.year < 1989]
        if est == '06008005-4':
            q_vic = q_vic.loc[(q_vic.index.year < 2005) & (q_vic.index > '1991-03-31')]
        
        q_vic.loc[q_vic.index.year == 1983] = np.nan
        q_Rapel_mon.loc[q_vic.index, est] = q_vic.iloc[:,0]
    
    # el 1997 del VIC da mal
    q_Rapel_mon.loc[q_Rapel_mon.index.year.isin([1997,1998]), estaciones+['06008005-4']] = np.nan
    
    # el 1997 del VIC da mal
    q_Rapel_mon.loc[q_Rapel_mon.index.year.isin([1981,1982]), '06006001-0'] = np.nan
    
    q_Rapel_mon.to_csv('Q_mon_RR_flags.csv')

    '''
    # =================================
    #     naturalizacion Rio Teno 
    # =================================
    '''
    
    q_Mataquito_daily = pd.read_csv(ruta_Mataquito_daily, index_col = 0, parse_dates = True)
    q_Mataquito_mon = flags_mon(q_Mataquito_daily)
    
    # ======================================
    #      Curva de embalse El Planchón
    # ======================================
    
    def VolEmbalse(h):
        return -4.192631*1e-3*h**3.+3.0688103539*1e1*h**2.-7.4865392042665*1e4*h+6.087244666258110*1e7
        
    h_planchon = pd.read_csv(ruta_Planchon, index_col = 0, parse_dates = True)
    idx =  h_planchon.dropna().index
    
    for ind,row in h_planchon.iterrows():
        h_planchon.loc[ind,'Vol (HM3)'] = VolEmbalse(float(h_planchon.loc[ind,'Nivel embalse (m.s.n.m.)']))
    
    # ========================================
    #   Entregas El Planchon de CNR
    # ========================================
    
    dVdT_El_Planchon = h_planchon['Vol (HM3)'].diff()*1e6/86400   

    op_El_Planchon = pd.DataFrame([], index = pd.date_range('01-01-1979','01-01-2021',freq = '1d'), columns = ['Q'])
    op_El_Planchon.loc[dVdT_El_Planchon.index,'Q'] = dVdT_El_Planchon.loc[dVdT_El_Planchon.dropna().index]
    op_El_Planchon = op_El_Planchon.loc[op_El_Planchon.index < '2020-04-01']
    op_El_Planchon = op_El_Planchon['Q']
    
    # ----Estaciones aguas abajo embalse
    
    estaciones_Teno = ['07102005-3', '07104002-K']
    
    # ------Estaciones Río Claro
    
    estaciones_Claro = ['07103001-6']
    
    # ========================================
    #   Agregar operación del embalse
    # ========================================   
        
    q_Mataquito_daily_aux = q_Mataquito_daily.copy()
    q_Mataquito_daily[estaciones_Teno] = np.nan
        
    for est in estaciones_Teno:
        idx = op_El_Planchon[op_El_Planchon.notnull()].index.intersection(q_Mataquito_daily_aux[est][q_Mataquito_daily_aux[est].notnull()].index)
        q_Mataquito_daily.loc[idx,est] = q_Mataquito_daily_aux.loc[idx,est]+op_El_Planchon.loc[idx]
    
    q_Mataquito_daily_mean = flags_mon(q_Mataquito_daily)
    
    ## ----mensual

    q_El_Planchon_mon = pd.read_excel(ruta_Planchon_CNR, sheet_name = 'datos', parse_dates = True, index_col = 0)
    q_Mataquito_mon_aux = q_Mataquito_mon.copy()
    q_Mataquito_mon[estaciones_Teno] = np.nan

    for est in estaciones_Teno:
       idx = q_El_Planchon_mon[q_El_Planchon_mon.notnull()].index.intersection(q_Mataquito_mon_aux[est][q_Mataquito_mon_aux[est].notnull()].index)
       q_Mataquito_mon.loc[idx,est] = q_Mataquito_mon_aux.loc[idx,est] - q_El_Planchon_mon.loc[idx,'Q']
       idx2 = q_Mataquito_daily_mean[est][q_Mataquito_daily_mean[est].notnull()].index
       q_Mataquito_mon.loc[idx2,est] = q_Mataquito_daily_mean.loc[idx2,est].values
       
      # recuperar caudales observados en temporada no de riego
    q_Mataquito_mon_aux = flags_mon(q_Mataquito_daily_aux)
    for est in estaciones_Teno:
        idx = q_Mataquito_mon_aux[est][q_Mataquito_mon_aux[est].notnull()].index.intersection(q_Mataquito_mon[est][q_Mataquito_mon[est].isnull()].index)
        q_Mataquito_mon.loc[idx,est] = q_Mataquito_mon_aux.loc[idx,est].values

       
    # ===================================
    #     Estero Upeo en Upeo
    # ===================================

    # Q_EsteroUpeoUpeo = q_Mataquito_mon['07116001-7'].copy().dropna()    
    
    # ========================================
    #   Sumar derechos aguas arriba
    # ========================================   
    
    # ---------Cargar derechos
    
    DAA_Mataquito = geopandas.read_file(ruta_DAA_Maule)
    DAA_Mataquito = DAA_Mataquito[DAA_Mataquito['Region'] == 7]
    DAA_Mataquito = DAA_Mataquito[(DAA_Mataquito['Tipo_Derec'] == 'Consuntivo') & (DAA_Mataquito['Ejercici_2'] == 'Continuo')]
    
    for ind,col in q_Mataquito_mon.iterrows():
        DAA_fecha = DAA_Mataquito.copy()[pd.to_datetime(DAA_Mataquito['Fecha_de_R']) < ind]
        
        DAA_fecha_RTBQLI  = DAA_fecha.copy()[DAA_fecha['Estacion'] == '07102005-3']
        DAA_fecha_RCLQ  = DAA_fecha.copy()[(DAA_fecha['Estacion'] == '07103001-6')]
        DAA_fecha_RTDJC  = DAA_fecha.copy()[(DAA_fecha['Estacion'] == '07102005-3') | (DAA_fecha['Estacion'] == '07102001-0')
                                            | (DAA_fecha['Estacion'] == '07103001-6') | (DAA_fecha['Estacion'] == '07104002-K')]
        DAA_fecha_EUEU  = DAA_fecha.copy()[DAA_fecha['Estacion'] == '07116001-7']

        # ----------- sumar por mes
        mes_DAA = list(dicc_DAA.keys())[list(dicc_DAA.values())[ind.month-1]-1]
        
        q_derechos_RTBQLI = DAA_fecha_RTBQLI[mes_DAA].dropna().sum()/1000.
        q_derechos_RCLQ = DAA_fecha_RCLQ[mes_DAA].dropna().sum()/1000.
        q_derechos_RTDJC = DAA_fecha_RTDJC[mes_DAA].dropna().sum()/1000.
        q_derechos_EUEU = DAA_fecha_EUEU[mes_DAA].dropna().sum()/1000.

        q_Mataquito_mon.loc[ind,'07102005-3'] += q_derechos_RTBQLI
        q_Mataquito_mon.loc[ind,'07103001-6'] += q_derechos_RCLQ
        q_Mataquito_mon.loc[ind,'07104002-K'] += q_derechos_RTDJC
        q_Mataquito_mon.loc[ind,'07116001-7'] += q_derechos_EUEU
        
    # ---------Trasvase Teno Chimbarrongo 
    
    q_CTK13 = flags_mon(pd.read_csv(ruta_CS, parse_dates = True, index_col = 0)['06033009-3'])
    
    idx = q_CTK13[q_CTK13.notnull()].index
    
    q_Mataquito_mon_copy = q_Mataquito_mon.copy()
    
    q_Mataquito_mon['07121006-5'] = np.nan
    q_Mataquito_mon['07123001-5 '] = np.nan

    q_Mataquito_mon.loc[idx,'07121006-5'] = q_Mataquito_mon_copy.loc[idx,'07121006-5']+q_CTK13
    q_Mataquito_mon.loc[idx,'07123001-5'] = q_Mataquito_mon_copy.loc[idx,'07123001-5']+q_CTK13
   

    # ---------Completar con VIC el rio Teno y el valle

    q_Mataquito_VIC_mon = pd.read_csv(ruta_Mataquito_VIC_mon, index_col = 0, parse_dates = True)
    
    # estaciones = ['07104001-1','07121006-5','07102001-0','07102005-3','07104002-K','07121011-1','07103001-6','07123001-5']
    estaciones = ['07121006-5','07123001-5']

    q_Mataquito_mon[['07372001-K','07121006-5','07123001-5']] = np.nan
    
    for est in estaciones:
        idx = q_Mataquito_mon[est][q_Mataquito_mon[est].isnull()].index
        idx = idx[idx.isin(q_Mataquito_VIC_mon.index)]
        q_Mataquito_mon.loc[idx, est] = q_Mataquito_VIC_mon.loc[idx,'Salida_'+est[1:-2]]
    
    q_Mataquito_mon.to_csv('Q_mon_Mataquito_flags.csv')

    
    '''    
    #==================================================================
    #   Naturalización del caudal de la cuenca del Río Maipo
    #==================================================================
    '''
    
    q_Maipo_day = pd.read_excel(ruta_Maipo_day, parse_dates = True, index_col = 0, sheet_name = 'data')
    q_MLA = pd.DataFrame(q_Maipo_day['05722002-3'])
    
    # La central hidroeléctrica El Arrayán restituye aguas arriba de Mapocho en Los Almentros
    # Fuente: https://www.gestionhidricamapochoalto.cl/maps/fullscreen/3/

    '''Derechos
    RIO MAPOCHO EN LOS ALMENDROS: 05722002-3
    ESTERO YERBA LOCA ANTES JUNTA SAN FRANCISCO: 05721001-K
    ESTERO ARRAYAN EN LA MONTOSA: 05722001-5
    '''
    
    # cargar derechos y subcuenca
    DAA = geopandas.read_file(ruta_DAA)
    cuenca_MLAL = geopandas.read_file(ruta_basins)
    cuenca_MLAL = cuenca_MLAL[cuenca_MLAL['gauge_id'] == 5722002]
    cuenca_MLAL.set_crs(epsg = 4326)
    cuenca_MLAL.to_crs(epsg = 32719, inplace = True)
    
    # seleccionar derechos de la subcuenca
    DAA_MLAL = geopandas.clip(DAA, cuenca_MLAL)   
    DAA_MLAL = DAA_MLAL[(DAA_MLAL['Tipo_Derec'] == 'Consuntivo') & (DAA_MLAL['Ejercici_2'] == 'Continuo')]
    
    for ind,col in q_MLA.iterrows():
        mes_DAA = list(dicc_DAA.keys())[ind.month-1]
        DAA_fecha = DAA_MLAL.copy()[pd.to_datetime(DAA_MLAL['Fecha_de_R']) < ind]

        q_derechos_MLA = DAA_fecha.copy()[mes_DAA].dropna().sum()/1000.

        if ind.year > 2011:
            q_derechos_MLA = DAA_fecha.copy()[mes_DAA].dropna().replace(500,750).sum()/1000. #Posterior a 2011 Anglo utiliza los derechos eventuales
        
        print(q_derechos_MLA)
        q_MLA.loc[ind] += q_derechos_MLA
    
    q_MLA = q_MLA[q_MLA.index >= '2000-01-01']
    q_MLA.to_csv(ruta_out+'\\QMLA_NAT.csv')
    
    # completar con VIC
    estaciones = ['05722002-3','05721001-K','05722001-5']
    for est in estaciones:
        q_vic = completarVIC(ruta_Maipo_VIC_mon, q_Maipo_mon[est], est)
        q_Maipo_mon.loc[q_vic.index, est] = q_vic.iloc[:,0]

    q_Maipo_mon.to_csv('RIO MAIPO_Q_mensual_Mapocho_RN.csv')
    
    #=========================================================================
    # Cálculo de la demanda hidroeléctrica y naturalización de caudales 
    #=========================================================================

    # RIO VOLCAN EN QUELTEHUES	05702001-6
    # RIO MAIPO EN LAS MELOSAS	05701002-9
    # RIO MAIPO EN EL MANZANO	05710001-K
    # RIO MAIPO EN SAN ALFONSO	05704002-5
    # RIO OLIVARES ANTES JUNTA RIO COLORADO	05706001-8
    # RIO COLORADO ANTES JUNTA RIO OLIVARES	05705001-2
    # RIO COLORADO ANTES JUNTA RIO MAIPO	05707002-1
    # ESTERO PUANGUE EN BOQUERON 05741001-9
    

    #####################################################################
    ### Estaciones donde se sumarán las extracciones hidroeléctricas  ###
    #####################################################################    
    
    q_Maipo_mon = pd.read_csv(ruta_Maipo_mon_Mapocho, index_col = 0, parse_dates = True)
    q_RVQ = q_Maipo_mon['05702001-6'].copy()
    q_RCJM = q_Maipo_mon['05707002-1'].copy()
    q_RCJO = q_Maipo_mon['05705001-2'].copy()
    q_ROJC = q_Maipo_mon['05706001-8'].copy()
    q_RMLM = q_Maipo_mon['05701002-9'].copy()


    gen = pd.read_excel(ruta_gen)
    sheets_dict = pd.read_excel(ruta_gen, sheet_name=None)
    
    
    dicc_etah = {
    

    ##################################
    ### Centrales R. Metropolitana ###
    ##################################

    
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
    'El Arrayán' : [.835, 73.5],
    

    ###########################
    ###   Centrales Rapel   ###
    ###########################
    
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
    'ech-la-compania-ii' : [.92, 43], #Turbinas Francis, Fuente: DIA
    
    ###########################
    ### Centrales Mataquito ###
    ###########################   
    
    'La Montaña 1' : [.91, 265],
    'La Montaña 2' : [.91, 265],

    #######################
    ### Centrales Maule ###
    #######################    
    
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
	
    indice = pd.to_datetime([])
    caudales_gen = pd.DataFrame([])
    
    for name, sheet in sheets_dict.items():
        if (any(char.isdigit() for char in name)) & ('sing' not in name):
            print(name)
            hoja2 = sheet.iloc[9:,:]
            hoja2.columns = hoja2.iloc[0]
            hoja2 = hoja2.drop(hoja2.index[0])
            hoja2 = hoja2.loc[:, hoja2.columns.notnull()]
            hoja2[hoja2['CÓDIGO CENTRAL'] == 'Total Anual'].index
            hoja2 = hoja2.loc[0:int(hoja2[hoja2['CÓDIGO CENTRAL'] == 'Total Anual'].index.values)-1,:]
            hoja2.index = pd.to_datetime(hoja2['CÓDIGO CENTRAL'])
            ### Falta dividir dias eta y h
            hoja2 = hoja2.resample('MS').sum()*3600*1e6/86400/9810 
            ###Dividir por días del mes
            hoja2 = hoja2.div(hoja2.index.daysinmonth.values, axis = 'index') #dividir por dias del mes
     
            ###########################
            ### Dividir por eta y h ###
            ###########################

            for est in dicc_etah.keys():
                if est in hoja2.columns:
                    hoja2[est] = hoja2[est]/np.prod(dicc_etah[est])
            
            hoja2 = hoja2[set(dicc_etah.keys()) & set(hoja2.columns)]
            indice = indice.append(hoja2.index)
            caudales_gen = pd.concat([caudales_gen,hoja2], axis=0, sort=False).fillna(0)
               
    for col in caudales_gen.columns:
        caudales_gen_2021, caudales_gen_2021_ci = pronostico_ARMA(caudales_gen[col].loc[caudales_gen[col].index <= '2021-01-01'],11, (1,1,60)) #1,1,18
        caudales_gen_2021[caudales_gen_2021 < 0] = 0
        caudales_gen_2021_ci[caudales_gen_2021_ci < 0] = 0
        
        ###############################
        ### Intervalos de confianza ###
        ###############################
        
        caudales_gen_2021_ci_lower = caudales_gen_2021_ci.iloc[:,0]
        caudales_gen_2021_ci_upper = caudales_gen_2021_ci.iloc[:,1]
        caudales_gen[col] = pd.concat([caudales_gen[col].loc[caudales_gen[col].index <= '2021-01-01'],caudales_gen_2021], axis=0, sort=False).fillna(0)
    
    caudales_gen.plot(legend = False)  
    caudales_gen.to_csv(r'.\caudales_generacion_hidro.csv')      

    caudales_gen_aux = caudales_gen.copy().loc[caudales_gen.index <= max(q_RVQ.index)]
    
    ratio_DAA_Olivares_Colorado = pd.read_excel(ruta_DAA_Olivares_Colorado, sheet_name = 'DAA', index_col = 0)
    
    #Repartir extracciones segun derechos
    q_bocatoma_Colorado = pd.DataFrame(caudales_gen_aux.loc[caudales_gen_aux.index]['ALFALFAL'].values*(1-ratio_DAA_Olivares_Colorado.loc[caudales_gen_aux.loc[caudales_gen_aux.index]['ALFALFAL'].index.month].values[:,0]), columns=['ALFALFAL'], index=caudales_gen_aux.index)
    q_bocatoma_Olivares = pd.DataFrame(caudales_gen_aux.loc[caudales_gen_aux.index]['ALFALFAL'].values*(ratio_DAA_Olivares_Colorado.loc[caudales_gen_aux.loc[caudales_gen_aux.index]['ALFALFAL'].index.month].values[:,0]), columns=['ALFALFAL'], index=caudales_gen_aux.index)
    

    #########################################################################
    ###      Sumar las extracciones aguas arriba de hidroeléctricas       ###
    #########################################################################
    
    q_RVQ.loc[caudales_gen_aux.index] = q_RVQ[caudales_gen_aux.index]+caudales_gen_aux.loc[caudales_gen_aux.index]['VOLCAN']
    q_ROJC.loc[caudales_gen_aux.index] = q_ROJC[caudales_gen_aux.index]+q_bocatoma_Olivares['ALFALFAL']
    q_RCJO.loc[caudales_gen_aux.index] = q_RCJO[caudales_gen_aux.index]+q_bocatoma_Colorado['ALFALFAL']
    q_RMLM.loc[caudales_gen_aux.index] = q_RMLM[caudales_gen_aux.index]+caudales_gen_aux.loc[caudales_gen_aux.index]['QUELTEHUES']-caudales_gen_aux.loc[caudales_gen_aux.index]['VOLCAN']
    
    #Usar mediciones de las extracciones de la DGA cuando están disponibles   
    index_caQueltehues = q_Maipo_mon[q_Maipo_mon['05705002-0'].notnull()].index
    q_RMLM.loc[index_caQueltehues] = q_Maipo_mon['05701002-9'].copy().loc[index_caQueltehues]+q_Maipo_mon['05705002-0'].copy().loc[index_caQueltehues]
       
    #########################################################################
    ### Cálculo de la demanda termoeléctrica y naturalización de caudales ###
    #########################################################################
    
    '''
    Río Maipo: Candelaria, Candelaria 1, Candelaria 2, Candelaria Diesel, Candelaria GNL,CMPC Cordillera, CMPC Tissue,
    El Nogal, Estancilla, RENCA, NUEVA RENCA, Nueva Renca Diesel, Nueva Renca GNL, Nueva Renca FA_GLP, 
    Chorrillos, Sepultura, Ermitaño
    Río Rapel: Esperanza, COLIHUES
    Río Mataquito: Cem Bio Bio IFO, Cem Bio Bio DIESEL, Cem Bio Bio,Teno, Trapén, El Peñón, San Lorenzo,
    San Lorenzo de Diego de Almagro U1, San Lorenzo de Diego de Almagro U2, San Lorenzo de Diego de Almagro U3, 
    Zapallar, Chile Generacion, 
    Río Maule: Constitución 1, Constitución 2, San Gregorio-Linares, Raso Power
    '''
    
    ##########################################
    ### Vr y Vc centrales termoeléctricas ####
    ##########################################
        
    dicc_vol = {    
            
    ##################################
    ### Centrales R. Metropolitana ###
    ##################################
    
    'CMPC Cordillera' : [13.7, 0.7], 'CMPC Tissue' : [13.7, 0.7],
    'El Nogal' : [24.3, 1.2], 'Estancilla' : [24.3, 1.2], 'RENCA' : [27.65,1.4], 
    'NUEVA RENCA' : [13.7, 0.7], 'Nueva Renca Diesel' : [24.3, 1.2], 'Nueva Renca GNL' : [13.7, 0.7], 'Nueva Renca FA_GLP' : [13.7, 0.7], 
    'Chorrillos' : [24.3, 1.2], 'Sepultura' : [24.3, 1.2], 'Ermitaño' : [24.3, 1.2], 'Loma Los Colorados' : [31., 1.6], 
    'Loma Los Colorados II' : [31., 1.6], 'Santa Marta' : [31., 1.6], 'Trebal Mapocho' : [31., 1.6], 
    'El Campesino 1' : [31., 1.6],
    #edeuco' : [24.3,1.2],'
    
    ###########################
    ### Centrales O'Higgins ###
    ###########################
    
    'Candelaria' : [13.7, 0.7], 'Candelaria 1' : [24.3, 1.2], 'Candelaria 2' : [24.3, 1.2], 'Candelaria Diesel' : [24.3, 1.2], 
    'Candelaria GNL' : [13.7, 0.7], 'Esperanza' : [24.3,1.2], 'COLIHUES' : [24.3,1.2], 'Energía Pacífico' : [31., 1.6], 
    'Las Pampas' : [31., 1.6], 'Santa Irene' : [31., 1.6], 'Tamm' : [31., 1.6],
        
    #######################
    ### Centrales Maule ###
    #######################    
    
    'Cem Bio Bio IFO' : [24.3,1.2], 'Cem Bio Bio DIESEL' : [24.3,1.2], 'Cem Bio Bio' : [24.3,1.2], 'Teno' : [24.3,1.2], 
    'San Lorenzo' : [24.3,1.2],  'Zapallar' : [24.3,1.2],  'Chile Generacion' : [24.3,1.2], 'Constitución 1' : [24.3,1.2], 
    'Constitución 2' : [24.3,1.2], 'Raso Power' : [24.3,1.2], 'Licantén' : [31., 1.6], 
    'Licantén LN' : [31., 1.6], 'Viñales' : [31., 1.6], 'CELCO': [31., 1.6], 'San Gregorio-Linares' : [24.3, 1.2], 'Maule' : [24.3, 1.2],
    
    }

    q_termo_consumo = pd.DataFrame([])
    
    for name, sheet in sheets_dict.items():
        if (any(char.isdigit() for char in name)) & ('sing' not in name):
            print(name)
            sheet_n = sheet.iloc[9:,:]
            sheet_n.columns = sheet_n.iloc[0]
            sheet_n = sheet_n.drop(sheet_n.index[0])
            sheet_n = sheet_n.loc[:, sheet_n.columns.notnull()]
            sheet_n[sheet_n['CÓDIGO CENTRAL'] == 'Total Anual'].index
            sheet_n = sheet_n.loc[0:int(sheet_n[sheet_n['CÓDIGO CENTRAL'] == 'Total Anual'].index.values)-1,:]
            sheet_n.index = pd.to_datetime(sheet_n['CÓDIGO CENTRAL'])
            ### Falta dividir dias eta y h
            sheet_n = sheet_n.resample('MS').sum()/86400 #Pasar a s
            ###Dividir por días del mes
            sheet_n = sheet_n.div(sheet_n.index.daysinmonth.values, axis = 'index') #dividir por dias del mes
     
            ########################################
            ### Multiplicar por volumen unitario ###
            ########################################
            
            centrales_termo = set(dicc_vol.keys()) & set(sheet_n.columns)
            sheet_n = sheet_n[centrales_termo]
            
            for est in centrales_termo:
                sheet_n[est] = sheet_n[est]*dicc_vol[est][1]
            
            q_termo_consumo = pd.concat([q_termo_consumo,sheet_n], axis=0, sort=False).fillna(0)
               
    for col in q_termo_consumo.columns:
        q_termo_consumo_2021, q_termo_consumo_2021_ci = pronostico_ARMA(q_termo_consumo[col].loc[q_termo_consumo[col].index <= '2021-01-01'],11, (2,1,60)) #1,1,18
        q_termo_consumo_2021[q_termo_consumo_2021 < 0] = 0
        q_termo_consumo_2021_ci[q_termo_consumo_2021_ci < 0] = 0
        
        ###############################
        ### Intervalos de confianza ###
        ###############################
        
        q_termo_consumo_2021_ci_lower = q_termo_consumo_2021_ci.iloc[:,0]
        q_termo_consumo_ci_upper = q_termo_consumo_2021_ci.iloc[:,1]
        q_termo_consumo[col] = pd.concat([q_termo_consumo[col].loc[q_termo_consumo[col].index <= '2021-01-01'],q_termo_consumo_2021], axis=0, sort=False).fillna(0)
    
    q_termo_consumo.plot(legend = False)
    q_termo_consumo.to_csv('caudales_generacion_termo.csv')
     
    
    # =======================================================
    #       Asignar las extracciones hidroeléctricas     
    # =======================================================
    
    q_Maipo_mon['05702001-6'] = q_RVQ
    q_Maipo_mon['05706001-8'] = q_ROJC
    q_Maipo_mon['05705001-2'] = q_RCJO 
    q_Maipo_mon['05701002-9'] = q_RMLM 

    #========================================================
    #   Asignar las extracciones de riego y AP por derechos   
    #========================================================
    #Cargar DAA   

    DAA_Maipo = pd.read_csv(ruta_DAA_Maipo)
    DAA_Maipo = DAA_Maipo[(DAA_Maipo['Tipo_Derec'] == 'Consuntivo') & (DAA_Maipo['Ejercici_2'] == 'Continuo')]
    
    DAA_Maule = geopandas.read_file(ruta_DAA_Maule)
    DAA_Maule = DAA_Maule[DAA_Maule['Region'] == 5]
    DAA_Maule = DAA_Maule[(DAA_Maule['Tipo_Derec'] == 'Consuntivo') & (DAA_Maule['Ejercici_2'] == 'Continuo')]
              
    for ind,col in q_Maipo_mon.iterrows():
        DAA_fecha = DAA_Maipo.copy()[pd.to_datetime(DAA_Maipo['Fecha_de_R']) < ind]
        DAA_fecha_2 = DAA_Maule.copy()[pd.to_datetime(DAA_Maule['Fecha_de_R']) < ind]

        DAA_fecha_RVQ  = DAA_fecha.copy()[DAA_fecha['Estacion'] == '05702001-6']
        DAA_fecha_ROJC = DAA_fecha.copy()[DAA_fecha['Estacion'] == '05706001-8']
        DAA_fecha_RCJO = DAA_fecha.copy()[(DAA_fecha['Estacion'] == '05705001-2') | (DAA_fecha['Estacion'] == '05705001-2')]
        DAA_fecha_RCJM = DAA_fecha.copy()[(DAA_fecha['Estacion'] == '05707002-1') | (DAA_fecha['Estacion'] == '05705001-2') | (DAA_fecha['Estacion'] == '05706001-8')]
        DAA_fecha_RMLM = DAA_fecha.copy()[(DAA_fecha['Estacion'] == '05701002-9') | (DAA_fecha['Estacion'] == '05701001-0')]
        DAA_fecha_RMLH = DAA_fecha.copy()[DAA_fecha['Estacion'] == '05701001-0']
        
        DAA_fecha_Colina_Peldehue = DAA_fecha_2.copy()[DAA_fecha_2['Estacion'] == '05735001-6']
        DAA_fecha_Quebrada_Ramon = DAA_fecha_2.copy()[DAA_fecha_2['Estacion'] == '05730008-6']
        DAA_fecha_EPEB = DAA_fecha.copy()[DAA_fecha['Estacion'] == '05741001-9']
        
        
        # En lugar del mes se debe usar el derecho promedio, por las pifias del CPA DGA
        mes_DAA = 'Caudal__l_'
        mes_DAA = list(dicc_DAA.keys())[list(dicc_DAA.values())[ind.month-1]-1]
        
        q_derechos_RVQ = DAA_fecha_RVQ[mes_DAA].dropna().sum()/1000.
        q_derechos_ROJC = DAA_fecha_ROJC[mes_DAA].dropna().sum()/1000.
        q_derechos_RCJO = DAA_fecha_RCJO[mes_DAA].dropna().sum()/1000.
        q_derechos_RCJM = DAA_fecha_RCJM[mes_DAA].dropna().sum()/1000.
        q_derechos_RMLM = DAA_fecha_RMLM[mes_DAA].dropna().sum()/1000.
        q_derechos_RMLH = DAA_fecha_RMLH[mes_DAA].dropna().sum()/1000.
        q_derechos_RCP = DAA_fecha_Colina_Peldehue[mes_DAA].dropna().sum()/1000.
        q_derechos_QR = DAA_fecha_Quebrada_Ramon[mes_DAA].dropna().sum()/1000.
        q_derechos_EPEB = DAA_fecha_EPEB[mes_DAA].dropna().sum()/1000.

        print(q_derechos_RVQ, q_derechos_ROJC,q_derechos_RCJO, q_derechos_RCJM, q_derechos_RMLM, q_derechos_RMLH)

        q_Maipo_mon.loc[ind,'05702001-6'] += q_derechos_RVQ
        q_Maipo_mon.loc[ind,'05706001-8'] += q_derechos_ROJC
        q_Maipo_mon.loc[ind,'05705001-2'] += q_derechos_RCJO
        q_Maipo_mon.loc[ind,'05705001-2'] += q_derechos_RCJM
        q_Maipo_mon.loc[ind,'05701002-9'] += q_derechos_RMLM
        q_Maipo_mon.loc[ind,'05701001-0'] += q_derechos_RMLH
        q_Maipo_mon.loc[ind,'05735001-6'] += q_derechos_RCP
        q_Maipo_mon.loc[ind,'05730008-6'] += q_derechos_QR
        q_Maipo_mon.loc[ind,'05741001-9'] += q_derechos_EPEB
     
              
    #====================================
    # Borrar los caudales observados  
    #====================================
    
    #Respaldar
    q_Maipo_mon_aux = q_Maipo_mon.copy()
    
    #Borrar los caudales observados intervenidos
    indice = indice[indice <= max(q_RVQ.index)]
    est_NAT = ['05702001-6', '05710001-K', '05704002-5', '05706001-8', '05705001-2','05701002-9',
               '05741001-9', '05735001-6', '05730008-6']
    q_Maipo_mon[est_NAT] = np.nan
    
    #############################################
    # Reemplazar por los cálculos de generación #
    #############################################
    
    indice = caudales_gen_aux['VOLCAN'][caudales_gen_aux['VOLCAN'].notnull()].index
    q_Maipo_mon['05702001-6'].loc[indice] = q_Maipo_mon_aux['05702001-6'].loc[indice] #RIO VOLCAN EN QUELTEHUES
    
    indice = caudales_gen_aux['ALFALFAL'][caudales_gen_aux['ALFALFAL'].notnull()].index #Se prorrateó
    q_Maipo_mon['05706001-8'].loc[indice] = q_Maipo_mon_aux['05706001-8'].loc[indice] #RIO OLIVARES ANTES JUNTA RIO COLORADO
    q_Maipo_mon['05705001-2'].loc[indice] = q_Maipo_mon_aux['05705001-2'].loc[indice] #RIO COLORADO ANTES JUNTA RIO OLIVARES
    
    indice = caudales_gen_aux['QUELTEHUES'][caudales_gen_aux['QUELTEHUES'].notnull()].index #Se calculó según Queltehues y Volcán
    q_Maipo_mon['05701002-9'].loc[indice] = q_Maipo_mon_aux['05701002-9'].loc[indice] #RIO MAIPO EN LAS MELOSAS
    
    indice = pd.date_range('1979-01-01','2020-12-01', freq = 'MS')
    q_Maipo_mon['05741001-9'].loc[indice] = q_Maipo_mon_aux['05741001-9']
    q_Maipo_mon['05735001-6'].loc[indice] = q_Maipo_mon_aux['05735001-6']
    q_Maipo_mon['05730008-6'].loc[indice] = q_Maipo_mon_aux['05730008-6']
  
    #############################################
    #  Reemplazar por los valores de J. McPhee  #
    #############################################
    
    q_Volcan_Quelethues_RN = pd.read_excel(ruta_VQ_RN, parse_dates = True, index_col = 0)
    q_Maipo_mon['05702001-6'].loc[q_Volcan_Quelethues_RN.index] = q_Volcan_Quelethues_RN.loc[q_Volcan_Quelethues_RN.index]['Q']
   
    q_MM_RN = pd.read_excel(ruta_MM_RN, parse_dates = True, index_col = 0)
    q_MSA_RN = pd.read_excel(ruta_MSA_RN, parse_dates = True, index_col = 0)
    q_ORC_RN = pd.read_excel(ruta_ORC_RN, parse_dates = True, index_col = 0)
    q_CRO_RN = pd.read_excel(ruta_CRO_RN, parse_dates = True, index_col = 0)
    q_MLM_RN = pd.read_excel(ruta_RMLM_RN, parse_dates = True, index_col = 0)
    q_RCJM_RN = pd.read_excel(ruta_CAJM_RN, parse_dates = True, index_col = 0)
    q_RCJM_RN = q_RCJM_RN.loc[q_RCJM_RN.index > '1979-03-01']
    
    ################################################
    # Escribir los valores conocidos de QRN en csv #
    ################################################

    q_Maipo_mon['05710001-K'].loc[q_MM_RN.index] = q_MM_RN.loc[q_MM_RN.index]['Q']
    q_Maipo_mon['05704002-5'].loc[q_MSA_RN.index] = q_MSA_RN.loc[q_MSA_RN.index]['Q']
    q_Maipo_mon['05706001-8'].loc[q_ORC_RN.index] = q_ORC_RN.loc[q_ORC_RN.index]['Q']
    q_Maipo_mon['05705001-2'].loc[q_CRO_RN.index] = q_CRO_RN.loc[q_CRO_RN.index]['Q']
    q_Maipo_mon['05707002-1'].loc[q_RCJM_RN.index] = q_RCJM_RN.loc[q_RCJM_RN.index]['Q']
    q_Maipo_mon['05701002-9'].loc[q_MLM_RN.index] = q_MLM_RN.loc[q_MLM_RN.index]['QMANAT']
    
    for est in est_NAT + ['05701002-9','05701001-0']:
        q_vic = completarVIC(ruta_Maipo_VIC_mon, q_Maipo_mon[est], est)
        q_Maipo_mon.loc[q_vic.index, est] = q_vic.iloc[:,0]
    
    q_Maipo_mon.to_csv('RIO MAIPO_Q_mensual_Mapocho_Manzano_RN_flags.csv')


    ''' 
    # ========================================
    #  Naturalización Rio Maule    
    # ========================================
    '''
    
    #Volumen embalsado Melado la última columna
    V_melado = pd.read_csv(ruta_melado, sep = ';', index_col = 2, parse_dates = True, dayfirst=True)
    V_melado = V_melado.loc[V_melado.iloc[:,1] == 'EMBALSE MELADO' ].iloc[:,-1]
    V_melado[V_melado <= 0] = np.nan
    dVdT_melado = V_melado.diff()*1e6/86400
    dVdT_melado = dVdT_melado[np.abs(dVdT_melado-dVdT_melado.mean()) <= 3.*dVdT_melado.std()]
    dVdT_melado = dVdT_melado.rename_axis('')
    
    plt.close("all")
    #Volúmenes de embalse Laguna del Maule y Laguna de La Invernada
    LMLI = pd.DataFrame(columns = ['Nro.', 'Fecha-Hora de Medicion',
           'Lagu Maule Volumen de Lago (Mill.m3)Min',
           'Lagu Maule Volumen de Lago (Mill.m3)Max',
           'Lagu Maule Volumen de Lago (Mill.m3)Media',
           'Embalse La IVolumen de Lago (Mill.m3)Min',
           'Embalse La IVolumen de Lago (Mill.m3)Max',
           'Embalse La IVolumen de Lago (Mill.m3)Media'])
    for filename in os.listdir(ruta_Vols):
        if filename.endswith(".xls"): 
            vols_aux = pd.read_html(ruta_Vols+'\\'+filename)[0]
    
            if filename in 'descarga_2008.xls':
                vols_aux = vols_aux.iloc[:,[0,1,5,6,7,11,12,13]]
                
            LMLI = pd.DataFrame(np.concatenate([LMLI.values,vols_aux.values]))
    
            continue
        else:
            continue
        
    LMLI.index = pd.to_datetime(LMLI[1], dayfirst = True)
    fechas = pd.date_range(freq='1d', start = min(LMLI.index), end = max(LMLI.index))
    dVdT_LMLI = pd.DataFrame(index = fechas, columns = [4,7])
    
    #Laguna del Maule y Laguna La Invernada
    LMLI_aux = LMLI[[4,7]]
    for i in range(len(LMLI_aux)-1):
        if np.abs(LMLI_aux.iloc[i+1,0]-LMLI_aux.iloc[i,0])/LMLI_aux.iloc[i,0] > 0.4:
            LMLI_aux.loc[LMLI_aux.index[i+1],4] = LMLI_aux.iloc[i,0]
    
    LMLI_aux.plot()
    
    #Afluentes - Entregas de Laguna del Maule y Laguna La Invernada
    dVdT_LMLI.loc[LMLI_aux.index,[4,7]] = LMLI_aux
    dVdT_LMLI = dVdT_LMLI.diff()*1e3/86400
    
    #Remover Outliers de DGA pifias
    dVdT_LMLI = dVdT_LMLI[np.abs(dVdT_LMLI-dVdT_LMLI.mean()) <= [1,1]*dVdT_LMLI.std()]
    dVdT_LMLI.plot()
        
    ''' 
    ### Estaciones ###
    #Maule en Armerillo 07321002-K
    #Canal Las Garzas 07308003-7 - en canales
    #Melado en Los Hierros 07317002-8 - en canales
    #Maule Norte Aforador 07321003-8 - en canales
    #Canal de Evacuación Central Pehuenche 07321005-4 - en canales
    #RIO MAULE EN DESAGUE LAGUNA DEL MAULE 07300001-7 - en canales
    #RIO CIPRESES EN DASAGUE LAGUNA LA INVERNADA 07306001-K - en canales
    '''     
    estaciones = ['07308003-7', '07317002-8', '07321003-8', '07321005-4', '07300001-7', '07306001-K']
    Q_canales = pd.read_csv(ruta_Maule_original, index_col = 0, parse_dates = True)[estaciones]
    mask_Pehuenche = Q_canales[Q_canales['07321005-4'].isna()].index
    
    #Pehuenche puesta en servicio 1991
    mask_Pehuenche = mask_Pehuenche[(mask_Pehuenche >= '2002-09-26') & (mask_Pehuenche <= '2020-04-17')]
    
    #Caudales Maule rellenados
    Q_MA = pd.read_csv(ruta_Maule_original, index_col = 0, parse_dates = True)['07321002-K']
    Q_MA_rec = pd.read_excel(ruta_MauleArmerillo_rec, index_col = 0, parse_dates = True)
    Q_MA_nat = Q_MA_rec.loc[Q_MA_rec.index] + pd.DataFrame(dVdT_LMLI.sum(axis = 1).loc[Q_MA_rec.index], columns = Q_MA_rec.columns)
    Q_MA_nat.plot()
    
    idx =  dVdT_melado.index.intersection(dVdT_LMLI[dVdT_LMLI.notnull()].index).intersection(Q_MA[Q_MA.notnull()].index).intersection(Q_canales.sum(axis = 1)[Q_canales.sum(axis = 1).notnull()].index)
    idx = idx[idx <= max(Q_MA.index)]
    Q_MA_nat2 = pd.DataFrame(index = pd.date_range(freq='1d', start = min(idx), end = max(idx)), columns = ['Q Maule Armerillo natural'])
    
    #Caudal Maule Armerillo Régimen Natural
    Q_MA_nat2.loc[idx] = pd.DataFrame(Q_MA.loc[idx].values + Q_canales.sum(axis = 1).loc[idx].values + dVdT_melado.loc[idx].values + dVdT_LMLI.sum(axis = 1).loc[idx].values, index = idx)
    Q_MA_nat2.loc[Q_MA_nat.dropna().index] = Q_MA_nat.dropna().values
    Q_MA_nat2[Q_MA_nat2 < 0] = 0
    Q_MA_nat2.plot()
    Q_MA_nat2.loc[mask_Pehuenche] = np.nan  
      
    Q_MA_nat2.plot(legend = False)
    Q_MA.loc[idx].plot()
    plt.ylabel('$Q m^3/s$')
    plt.legend(['río Maule en Armerillo en régimen natural','río Maule en Armerillo observado'])
    
    plt.figure()

    Q_MA_nat2_mon = flags_mon(Q_MA_nat2)
    Q_MA_nat_Carla = pd.read_csv(ruta_MA_RN)
    Q_MA_nat_DGA = pd.read_csv(ruta_MauleArmerillo_NAT_DGA, parse_dates = True, index_col = 0, sep = ';')
    Q_MA_nat_DGA = Q_MA_nat_DGA.loc[Q_MA_nat_DGA.index >= '1979-04-01']
    fechas = pd.date_range(freq='1MS', start = '1979-04-01', end = '2015-03-01')
    Q_MA_nat_mon = pd.DataFrame(np.interp(np.linspace(1979,2015,12*(2015-1979)),Q_MA_nat_Carla['month'],Q_MA_nat_Carla['Q']), index = fechas, columns = ['Q'])
    Q_MA_nat_mon.plot()
    Q_MA_nat_mon_vF = pd.DataFrame(index = pd.date_range(freq='1MS', start = '1979-04-01', end = '2020-04-01'), columns = ['Q'])
    Q_MA_nat_mon_vF.loc[Q_MA_nat2_mon.dropna().index] = Q_MA_nat2_mon.dropna().values
    Q_MA_nat_mon_vF.loc[Q_MA_nat_mon.dropna().index] = Q_MA_nat_mon.dropna().values
    Q_MA_nat_mon_vF.loc[Q_MA_nat_DGA.dropna().index] = Q_MA_nat_DGA.dropna().values
    Q_MA_nat_mon_vF.to_csv('Q_mon_RMA_flags.csv')
    
    #Río Maule en Desagüe Laguna del Maule
    q_Laguna_Maule = pd.read_csv(ruta_Maule_original, index_col = 0, parse_dates = True)['07300001-7']
    idx = q_Laguna_Maule.index.intersection(dVdT_LMLI[4].dropna().index)
    q_Laguna_Maule_NAT = q_Laguna_Maule.loc[idx]+dVdT_LMLI[4].loc[idx]
    q_Laguna_Maule_NAT[q_Laguna_Maule_NAT < 0] = 0
    
    q_Laguna_Maule_NAT = flags_mon(q_Laguna_Maule_NAT)
    
    # # ------completar con VIC
    # q_Laguna_Maule_NAT = completarVIC(ruta_Maule_VIC_mon, q_Laguna_Maule_NAT, '07300001-7')
    # q_Laguna_Maule_NAT.to_csv('Q_mon_Laguna_Maule_flags.csv') 
    
    # ===================================
    #         Rio Melado en El Salto 
    # ===================================
    
    Q_Maule = pd.read_csv(ruta_Maule_original, index_col = 0, parse_dates = True)
    q_MeladoSalto = Q_Maule['07317005-2'].copy().dropna() 
    q_MeladoSalto = flags_mon(q_MeladoSalto)
    
    # ===================================
    #       Perquilauquen en San Manuel 
    # ===================================
    
    q_Perquilauquen = Q_Maule['07330001-0'].copy().dropna() 
    q_Perquilauquen = flags_mon(q_Perquilauquen)
            
    #Cargar DAA Maule  
    DAA_Maule = geopandas.read_file(ruta_DAA_Maule)
    DAA_Maule = DAA_Maule[(DAA_Maule['Region'] == 7) | (DAA_Maule['Region'] == 8)]
    DAA_Maule = DAA_Maule[(DAA_Maule['Tipo_Derec'] == 'Consuntivo') & (DAA_Maule['Ejercici_2'] == 'Continuo')]
          
    for ind in Q_Maule.index:
       DAA_fecha = DAA_Maule.copy()[pd.to_datetime(DAA_Maule['Fecha_de_R']) < ind]
       DAA_fecha_Melado = DAA_fecha.copy()[DAA_fecha['Estacion'] == '07317005-2']
       DAA_fecha_Perquilauquen = DAA_fecha.copy()[DAA_fecha['Estacion'] == '07330001-0']

     
       # En lugar del mes se debe usar el derecho promedio, por las pifias del CPA DGA
       q_derechos_MeladoSalto = DAA_fecha_Melado['Caudal__l_'].dropna().sum()/1000.
       q_derechos_Perquilauquen = DAA_fecha_Perquilauquen['Caudal__l_'].dropna().sum()/1000.


       if ind in q_MeladoSalto.index:
           q_MeladoSalto.loc[ind] += q_derechos_MeladoSalto
           
       if ind in q_Perquilauquen.index:
           q_Perquilauquen.loc[ind] += q_derechos_Perquilauquen

     # ==========================================
     #  Rio Melado en El Salto Régimen Natural
     # ==========================================
    idx = Q_Maule['07317002-8'][Q_Maule['07317002-8'].notnull()].index.intersection(q_MeladoSalto[q_MeladoSalto.notnull()].index)
    q_MeladoSalto_aux = q_MeladoSalto.copy()
    q_MeladoSalto.loc[q_MeladoSalto.index] = np.nan
    q_MeladoSalto.loc[idx] = q_MeladoSalto_aux.loc[idx]+Q_Maule.loc[idx,'07317002-8'].values
        
    # ---------completar con VIC
    # q_MeladoSalto = completarVIC(ruta_Maule_VIC_mon, q_MeladoSalto, '07317005-2')
    
    # ==========================================
    #       Rio Perquilauquen en San Manuel 
    # ==========================================
     
    # q_Perquilauquen = completarVIC(ruta_Maule_VIC_mon, q_Perquilauquen, '07330001-0')
    
    # ==================================
    #       naturalizacion Río Ancoa
    # ==================================

    #Cargar datos originales DGA
    Q_Maule = pd.read_csv(ruta_Maule_original, index_col = 0, parse_dates = True)
    # Ancoa en El Morro
    Q_Ancoa = Q_Maule['07355002-5'].copy().dropna() 
    # Canal Melado
    Q_Melado = Q_Maule['07317002-8'].copy().dropna() 
    # Canal Roblería
    Q_Robleria = Q_Maule['07355008-4'].copy().dropna() 
    # RIO ACHIBUENO EN LA RECOVA
    Q_Achibueno_Rec = Q_Maule['07354002-K'].copy().dropna()
    Q_Achibueno_Rec_NAT = Q_Achibueno_Rec.copy()
    #RIO ACHIBUENO EN LOS PEGNASCOS no tiene registros
     
    #Indices comunes
    idx = Q_Ancoa.index.intersection(Q_Melado.index) 
    idx = idx.intersection(Q_Robleria.index)
    idx = idx[idx < '2012-01-01'] 
    
    #Cargar DAA Maule  
    DAA_Maule = geopandas.read_file(ruta_DAA_Maule)
    DAA_Maule = DAA_Maule[(DAA_Maule['Region'] == 7) | (DAA_Maule['Region'] == 8)]
    DAA_Maule = DAA_Maule[(DAA_Maule['Tipo_Derec'] == 'Consuntivo') & (DAA_Maule['Ejercici_2'] == 'Continuo')]
         
    for ind,col in Q_Maule.iterrows():
       DAA_fecha = DAA_Maule.copy()[pd.to_datetime(DAA_Maule['Fecha_de_R']) < ind]
       DAA_fecha_Ancoa = DAA_fecha.copy()[(DAA_fecha['Estacion'] == '07355003-3') | (DAA_fecha['Estacion'] == '07355008-4')| (DAA_fecha['Estacion'] == '07355002-5')]
       DAA_fecha_Achibueno = DAA_fecha.copy()[(DAA_fecha['Estacion'] == '07354002-K') | (DAA_fecha['Estacion'] == '07354001-1')]
      
       # En lugar del mes se debe usar el derecho promedio, por las pifias del CPA DGA
       q_derechos_Ancoa = DAA_fecha_Ancoa['Caudal__l_'].dropna().sum()/1000.
       q_derechos_Achibueno = DAA_fecha_Achibueno['Caudal__l_'].dropna().sum()/1000.

       if ind in Q_Ancoa.index:
           Q_Ancoa.loc[ind] += q_derechos_Ancoa
       if ind  in Q_Achibueno_Rec_NAT.index:
           Q_Achibueno_Rec_NAT.loc[ind] += q_derechos_Ancoa

    # ===================================
    #  Ancoa en El Morro Régimen Natural
    # ===================================

    Q_Ancoa_NAT = Q_Ancoa.loc[idx] - Q_Melado.loc[idx] + Q_Robleria.loc[idx]
    Q_Ancoa_NAT[Q_Ancoa_NAT < 0] = 0
    Q_Ancoa_NAT = flags_mon(Q_Ancoa_NAT)
    
    # ---------completar con VIC
    Q_Ancoa_NAT = completarVIC(ruta_Maule_VIC_mon, Q_Ancoa_NAT, '07355002-5')
    
    Q_Ancoa_NAT.to_csv('Ancoa_en_El_Morro_NAT.csv') 
    
    # ===================================
    #  RIO ACHIBUENO EN LA RECOVA NAT
    # ===================================

    Q_Achibueno_Rec_NAT = flags_mon(Q_Achibueno_Rec_NAT)
    
    # ---------completar con VIC
    # Q_Achibueno_Rec_NAT = completarVIC(ruta_Maule_VIC_mon, Q_Achibueno_Rec_NAT, '07354002-K')
   
    Q_Achibueno_Rec_NAT.to_csv('Achibueno_en_La_Recova_NAT.csv') 
     
    # ==================================
    #       naturalizacion Longavi
    # ==================================
    
    #Cargar datos originales DGA
    Q_Maule = pd.read_csv(ruta_Maule_original, index_col = 0, parse_dates = True)
    
    # ===================================
    #      Longavi en la Quiriquina
    # ===================================

    Q_LongaviQuiri = Q_Maule['07350001-K'].copy().dropna() 
    Q_LongaviQuiri_NAT = Q_LongaviQuiri.copy()
    
    # ===================================
    #     RIO LONGAVI EN EL CASTILLO
    # ===================================

    Q_LongaviCastillo = Q_Maule['07350003-6'].copy().dropna() 
    Q_LongaviCastillo_NAT = Q_LongaviCastillo.copy()
    
    #Cargar DAA Maule  
    DAA_Maule = geopandas.read_file(ruta_DAA_Maule)
    DAA_Maule = DAA_Maule[(DAA_Maule['Region'] == 7) | (DAA_Maule['Region'] == 8)]
    DAA_Maule = DAA_Maule[(DAA_Maule['Tipo_Derec'] == 'Consuntivo') & (DAA_Maule['Ejercici_2'] == 'Continuo')]
  
    #Embalses
    V_Bullileo =  pd.read_csv(ruta_melado, sep = ';', index_col = 2, parse_dates = True, dayfirst=True)
    V_Bullileo = V_Bullileo.loc[V_Bullileo.iloc[:,1] == 'BULLILEO EMBALSE (Lago)' ].iloc[:,-1]
    V_Bullileo[V_Bullileo <= 0] = np.nan
    dVdT_Bullileo = V_Bullileo.diff()*1e6/86400
    dVdT_Bullileo = dVdT_Bullileo[np.abs(dVdT_Bullileo-dVdT_Bullileo.mean()) <= 2.*dVdT_Bullileo.std()]
    dVdT_Bullileo = dVdT_Bullileo.rename_axis('')
    dVdT_Bullileo.plot()
    
    for ind,col in Q_Maule.iterrows():
       DAA_fecha = DAA_Maule.copy()[pd.to_datetime(DAA_Maule['Fecha_de_R']) < ind]
       DAA_fecha_LongaviCast = DAA_fecha.copy()[DAA_fecha['Estacion'] == '07350003-6']
       DAA_fecha_LongaviQuiri = DAA_fecha.copy()[(DAA_fecha['Estacion'] == '07350003-6') | (DAA_fecha['Estacion'] == '07350001-K')]
      
       # En lugar del mes se debe usar el derecho promedio, por las pifias del CPA DGA
       q_derechos_LongaviCast = DAA_fecha_LongaviCast['Caudal__l_'].dropna().sum()/1000.
       q_derechos_LongaviQuiri = DAA_fecha_LongaviQuiri['Caudal__l_'].dropna().sum()/1000.

       if ind in Q_LongaviCastillo.index:
           Q_LongaviCastillo_NAT.loc[ind] += q_derechos_LongaviCast
       if ind in Q_LongaviQuiri.index:
           Q_LongaviQuiri_NAT.loc[ind] += q_derechos_LongaviQuiri
   
    Q_LongaviQuiri_NAT.loc[[x for x in Q_LongaviQuiri_NAT.index if x not in dVdT_Bullileo.index]] = np.nan
    Q_LongaviQuiri_NAT.loc[dVdT_Bullileo.index] = Q_LongaviQuiri_NAT.loc[dVdT_Bullileo.index].values + dVdT_Bullileo.values
    Q_LongaviQuiri_NAT[Q_LongaviQuiri_NAT < 0] = 0
     
     #Longavi en El Castillo Régimen Natural
    Q_LongaviCastillo_NAT = flags_mon(Q_LongaviCastillo_NAT)
     
    # ---------no completar con VIC, no está calibrado
    # Q_LongaviCastillo_NAT = completarVIC(ruta_Maule_VIC_mon, Q_LongaviCastillo_NAT, '07350003-6')
    Q_LongaviCastillo_NAT.to_csv('Rio_Longavi_en_El_Castillo_NAT.csv') 
         
    #Longavi en La Quiriquina Régimen Natural
    Q_LongaviQuiri_NAT = flags_mon(Q_LongaviQuiri_NAT)
    
    # ---------completar con VIC
    Q_LongaviQuiri_NAT = completarVIC(ruta_Maule_VIC_mon, Q_LongaviQuiri_NAT, '07350001-K')
    Q_LongaviQuiri_NAT.to_csv('Rio_Longavi_en_La_Quiriquina_NAT.csv') 
     
    '''
    # ==================================
    #       guardar Maule
    # ==================================
    '''
              
    q_mon = pd.read_excel(ruta_Maule_mon, parse_dates = True, index_col = 0, sheet_name  = 'data')
    q_flags = pd.read_excel(ruta_Maule_mon, parse_dates = True, index_col = 0, sheet_name  = 'info data')
    q_flags_aux = q_flags.copy()
    q_flags_aux = copyIndex(q_flags_aux)
    q_mon[q_flags < 20] = np.nan #Caudales mensuales con flags
    q_mon.drop(['Ano','Mes'], axis=1, inplace=True)
    q_mon['07321002-K'] = Q_MA_nat_mon_vF
    
    #Crear backup
    q_mon_Maule = q_mon.copy()
    
    #Borrar intervenidos
    #Laguna La Invernada, Maule en Los Baños, Ancoa en el Morro,Achibueno en la Recova, 
    #Laguna del Maule, Longavi en La Quiriquina, Longavi en El Castillo
    q_mon_Maule[['07306001-K','07303000-5', '07355002-5', '07354002-K',
                 '07300001-7','07350001-K','07350003-6','07317005-2']] = np.nan
    
    # ====================================
    # Guardar caudales naturalizados #
    # ====================================
        
    indice = caudales_gen.index.intersection(dVdT_LMLI[4].index) 
    idx = list(indice.intersection(q_mon['07303000-5'].dropna().index))
    q_mon_Maule.loc[idx,'07303000-5'] = pd.DataFrame(q_mon.loc[idx,'07303000-5'].values+caudales_gen.loc[idx,'Hidro La Mina'].values+dVdT_LMLI.loc[idx,4].values, index= idx).values
    q_mon_Maule['07303000-5'][q_mon_Maule['07303000-5'] < 0] = 0
    q_mon_Maule.loc[Q_Ancoa_NAT.index,'07355002-5'] = Q_Ancoa_NAT.values
    q_mon_Maule.loc[Q_Achibueno_Rec_NAT.index,'07354002-K'] = Q_Achibueno_Rec_NAT.values
    q_mon_Maule.loc[q_Laguna_Maule_NAT.index,'07300001-7'] = q_Laguna_Maule_NAT.values
    q_mon_Maule.loc[Q_LongaviQuiri_NAT.index,'07350001-K'] = Q_LongaviQuiri_NAT.values
    q_mon_Maule.loc[Q_LongaviCastillo_NAT.index,'07350003-6'] = Q_LongaviCastillo_NAT.values
    q_mon_Maule.loc[q_MeladoSalto.index,'07317005-2'] = q_MeladoSalto.values
    q_mon_Maule.loc[q_Perquilauquen.index,'07330001-0'] = q_Perquilauquen.values
    
    q_mon_Maule.to_csv('Q_mon_RM_flags.csv')
    
    
    '''
    # ==================================
    #       Cuenca del río Ñuble 
    # ==================================
    '''
    
    q_nuble = pd.read_excel(ruta_q_Nuble, index_col = 0, parse_dates = True, sheet_name = 'data')
    
    # sumar derechos aguas arriba
    
     #Cargar DAA Maule  
    DAA_Nuble = geopandas.read_file(ruta_DAA_Nuble)
    DAA_Nuble = DAA_Nuble[(DAA_Nuble['Tipo Derec'] == 'Consuntivo') & (DAA_Nuble['Ejercicio'] == 'Permanente y Continuo')]
         
    for ind,col in q_nuble.iterrows():
       DAA_fecha = DAA_Nuble.copy()[pd.to_datetime(DAA_Nuble['Fecha de R']) < ind]
       DAA_fecha_Nuble_SF_1 = DAA_fecha.copy()[(DAA_fecha['Estacion'] == '08106001-0') | (DAA_fecha['Estacion'] == '08106002-9')]
       DAA_fecha_Nuble_SF_2 = DAA_fecha.copy()[DAA_fecha['Estacion'] == '08106002-9']
      
       # En lugar del mes se debe usar el derecho promedio, por las pifias del CPA DGA
       q_derechos_Nuble_SF_1 = float(DAA_fecha_Nuble_SF_1['Caudal Anu'].str.replace(',','.').dropna().sum())/1000.
       q_derechos_Nuble_SF_2 = float(DAA_fecha_Nuble_SF_2['Caudal Anu'].str.replace(',','.').dropna().sum())/1000.

       if ind in q_nuble.index:
           q_nuble.loc[ind,'08106001-0'] += q_derechos_Nuble_SF_1
           q_nuble.loc[ind,'08106002-9'] += q_derechos_Nuble_SF_2
    
    q_nuble.to_csv('Q_mon_Nuble_NAT_flags.csv')

if __name__ == "__main__":
    main()