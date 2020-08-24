# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 21:48:38 2020

@author: Carlos
"""

import flopy
import os
import sys
import flopy.utils.binaryfile as bf
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from flopy.export import vtk


#%% Preproceso - construir el modelo
# Chequear instalación de floPy
try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join('..', '..'))
    sys.path.append(fpth)
    import flopy

os.chdir(r'D:\Curso Python v1.2.0\1. Curso Python\Flopy5')

#Fijar rutas
modelname = "modelo_real"
modelpath = "./Model"
# Creat objeto Modflow Nwt de la clase modflow
mf1 = flopy.modflow.Modflow(modelname, exe_name="./Exe/MODFLOW-NWT_64.exe", version="mfnwt",model_ws=modelpath)
nwt = flopy.modflow.ModflowNwt(mf1 ,maxiterout=150,linmeth=2)

#Raster de la geometría del modelo
demPath ="./Rst/DEM_200.tif"
crPath = "./Rst/CR.tif"

#Abrir rasters
demDs =gdal.Open(demPath)
crDs = gdal.Open(crPath)
geot = crDs.GetGeoTransform()

# Transformar las bandas de los rasters a arreglos
demData = demDs.GetRasterBand(1).ReadAsArray()
crData = crDs.GetRasterBand(1).ReadAsArray()

#Visualizar DEM
plt.imshow(demData)
plt.colorbar()
plt.show()

#Elevacion de las layers
 #7.5 m bajo la topografía
Layer1 = demData - 7.5
# 15m bajo la topografía
Layer1_2 = demData - 15 
Layer2 = (demData - 3000)*0.8 + 3000 #
Layer3 = (demData - 3000)*0.5 + 3000
Layer4 = (demData>0)*3000

#Doiscretización espacial y temporal 
ztop = demData #El top del layer 1 es la topografia
#Los bottom de cada layer son los tops del inferior
zbot = [Layer1, Layer1_2, Layer2, Layer3, Layer4]
# Número de layers
nlay = 5 
nrow = demDs.RasterYSize
ncol = demDs.RasterXSize
delr = geot[1]
delc = abs(geot[5])

#Crear paquetes de modflow
#Paquete DIS
dis = flopy.modflow.ModflowDis(mf1, nlay,nrow,ncol,delr=delr,delc=delc,top=ztop,botm=zbot,itmuni=1)

# Paquete BAS
iboundData = np.zeros(demData.shape, dtype=np.int32)
iboundData[crData > 0 ] = 1 #Los ibound = 1 son celdas activas, los ibound = 0 son inactivas y los ibound = -1 son condiciones de borde
bas = flopy.modflow.ModflowBas(mf1,ibound=iboundData,strt=ztop, hnoflo=-999.99)

#Lista de las conductividades hidráulicas
hk = [1E-4, 1E-5, 1E-7, 1E-8, 1E-9]

#Crear el paqeute Upstream Weighting para MF NWT
upw = flopy.modflow.ModflowUpw(mf1, laytyp = [1,1,1,1,0], hk = hk, ipakcb = 53) #Las primeras 4 layers son libres y la última es un acuífero confinado

#Crear un arreglo numérico para la recarga 
rch = np.ones((nrow, ncol), dtype=np.float32) * 0.120/365/86400
rch_data = {0: rch}
# Crear el paquete RCH del modelo MODFLOW NWT 
rch = flopy.modflow.ModflowRch(mf1, nrchop=3, rech =rch_data)

#Crear un arreglo numérico para la evapotranspiración 
evtr = np.ones((nrow, ncol), dtype=np.float32) * 1/365/86400
evtr_data = {0: evtr}
# Crear el paquete EVT del modelo MODFLOW NWT 
evt = flopy.modflow.ModflowEvt(mf1,nevtop=1,surf=ztop,evtr=evtr_data, exdp=0.5)

#Agregar drenes donde hay ríos en el raster
river = np.zeros(demData.shape, dtype=np.int32)
river[crData == 3 ] = 1
list = []
for i in range(river.shape[0]):
    for q in range(river.shape[1]):
        if river[i,q] == 1:
            list.append([0,i,q,ztop[i,q],0.001]) #layer,row,column,elevation(float), conductancia = 0.001
rivDrn = {0:list}

#Crear el paquete DRN
drn = flopy.modflow.ModflowDrn(mf1, stress_period_data=rivDrn)

#Definir los stress periods y variables del modelo a guardar

spd = {(0, 0): ['save head', 'save budget']}
oc = flopy.modflow.ModflowOc(mf1, stress_period_data=spd)
oc.get_budgetunit()

# Crear el paquete OC
#oc = flopy.modflow.ModflowOc(mf1) #ihedfm= 1, iddnfm=1

#Crear el .nam
mf1.write_input()

#Verificar el modelo
mf1.check()

#Correr el modelo
mf1.run_model()


#%% Postproceso
plt.close("all")

#Importar el modelo letendo el -nam 
ml = flopy.modflow.Modflow.load('./Model/'+modelname+'.nam')

#Graficamos llas celdas activas en planta
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
modelmap = flopy.plot.ModelMap(model=ml, rotation=0) #Modelmap crea el gráfico de nuestro pobjeto modelo ml
quadmesh = modelmap.plot_ibound(color_noflow='black') #plot_ibound grafica las celdas activas ibound = 1
linecollection = modelmap.plot_grid(linewidth=0.4) #Graficamos la grilla
plt.savefig('Celdas activas.png')

# Graficamos una sección del modelo con celdas activas
fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(1, 1, 1)
columna = 81
# Creamos un objeto de sección mediante la clase ModelCrossSection
modelxsect = flopy.plot.ModelCrossSection(model=ml, line={'Column': 80})
patches = modelxsect.plot_ibound(color_noflow='black')
linecollection = modelxsect.plot_grid(linewidth=0.4)
t = ax.set_title('Columna '+str(columna)+' de la Grilla')
plt.savefig('Seccion_columna_'+str(columna)+'.png')

#Graficamos los drenes
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
modelmap = flopy.plot.ModelMap(model=ml, rotation=0)
quadmesh = modelmap.plot_ibound(color_noflow='black')
quadmesh = modelmap.plot_bc('DRN', color='yellow')
linecollection = modelmap.plot_grid(linewidth=0.4)
plt.savefig('Drenes.png')

#Graficar los niveles del modelo
fname = os.path.join(modelpath, 'modelo_real.hds')
hdobj = flopy.utils.HeadFile(fname)
head = hdobj.get_data()

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
modelmap = flopy.plot.ModelMap(model=ml, rotation=0)
quadmesh = modelmap.plot_ibound()
quadmesh = modelmap.plot_array(head, masked_values=[-999.99], alpha=.99, cmap = 'Blues')
linecollection = modelmap.plot_grid(linewidth=0.2)
cb = plt.colorbar(quadmesh, shrink=0.5)
plt.xlabel('E (m)')
plt.ylabel('N (m)')
plt.show()
plt.savefig('niveles.png')

# Leer cbb y graficar direcciones de flujo
cbb = flopy.utils.CellBudgetFile('./Model/'+modelname+'.cbc')
frf = cbb.get_data(text='FLOW RIGHT FACE')[0]
fff = cbb.get_data(text='FLOW FRONT FACE')[0]
fig = plt.figure(figsize=(15, 15))

ax = fig.add_subplot(1, 1, 1, aspect='equal')
ax.set_title('Direcciones de flujo')
mapview = flopy.plot.PlotMapView(model=mf1)
quadmesh = mapview.plot_ibound()
quadmesh = mapview.plot_array(head, masked_values=[-999.99], alpha=1, cmap = 'Blues')
quiver = mapview.plot_discharge(frf, fff, head = head, normalize=True, scale=50)  # no head array for volumetric discharge
linecollection = modelmap.plot_grid(linewidth=0.9)
cb = plt.colorbar(quadmesh, shrink=0.5)
cb.ax.get_yaxis().labelpad = 15
cb.ax.set_ylabel('Nivel (m.s.n.m.)', rotation=270)

plt.xlabel('E (m)')
plt.ylabel('N (m)')
plt.show()
plt.savefig('Direcciones de flujo.png')


#Exportar propiedades hidráulicas a vtk
mf1.upw.export('upw.vtk',  fmt='vtk', point_scalars=True)

#Exportar heads a vtk
vtk.export_heads(mf1, fname, './heads', kstpkper=[(0,0)],
                 point_scalars=True, nanval=-999.99)
#Exportar drenes a vtk
mf1.drn.export('drain.vtk',  fmt='vtk', point_scalars=True)

#Exportar paquete  del modelo a shapefile
mf1.upw.export('modelo_real.shp')
