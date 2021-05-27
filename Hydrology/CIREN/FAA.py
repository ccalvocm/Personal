# Describir el codigo:

# 1. importar las capas que vamos a plotear
# 2. cargar las capas
# 3. plotear


# importar librerias
import os # la libreria os nos permite trabajar con la maquina
import geopandas # libreria para trabajar con capas
from matplotlib import pyplot
import contextily


# -------------------------------
# PASO 1 IMPORTAR LAS CAPAS
# -------------------------------

ruta = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH', 'Cuencas', 'Cuencas_DARH_2015_AOHIA_ZC.geojson')


# ------------------------------
# PASO 2 CARGAR LA CAPA
# -----------------------------

cuencas = geopandas.read_file(ruta)

print(cuencas.crs) # imprimir el sistema de referencia que trae la capa

# ------------------------------
# PASO 3 PLOTEAR LA CAPA
# ------------------------------

fig,ax = pyplot.subplots()

# agregar opciones al plot

# fc: facecolor --> color de relleno (poligono)
# ec: edgecolor --> color del borde (poligono)
# alpha: opacidad --> valor entre 0 y 1

# cambiemos el sistema coordenado (comentar/descomentar prox linea)
#cuencas = cuencas.to_crs('EPSG:4326') # 4326 es WGS 84 geografica
#print(cuencas.crs)

cuencas.plot(ax=ax, fc = 'none', ec = 'red')
contextily.add_basemap(ax=ax, crs='EPSG:32719')

# guardar figura
pyplot.savefig('FAA_plot.jpg', format='jpg')
pyplot.close()



