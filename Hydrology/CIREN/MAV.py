#Descrbir el código:

#Importar capas a plotear
#Cargar las capas
#Plotear

#Importar librerias

#---------------------------
#IMPORTAR CAPAS
#--------------------------

import os
import geopandas
from matplotlib import pyplot

import contextily

ruta = os.path.join("..","Etapa 1 y 2","GIS","Cuencas_DARH","Cuencas","Cuencas_DARH_2015_AOHIA_ZC.geojson")

#CARGAR LA CAPA


cuencas = geopandas.read_file(ruta)

print(cuencas.crs)

#PLOTEAR LA CAPA

fig, ax = pyplot.subplots()

#Agregar opciones al plot

# fc: facecolor --> color de relleno de poligono
# ec: edgecolor --> color del borde del poligono
#alpha: opacidad --> valor entre 0 y 1

#cambiar sistema de coordenadas

#cuencas = cuencas.to_crs("EPSG:4326") # 4326 es coord geografica
#print(cuencas.crs)

cuencas.plot(ax=ax, fc = "none", ec = "red")
contextily.add_basemap(ax=ax, crs="EPSG:32719")

#guardar figura

#pyplot.show()
pyplot.savefig("MAV_plot.jpg", format="jpg")
pyplot.close()


