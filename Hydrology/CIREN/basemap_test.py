import matplotlib
#matplotlib.use('Qt5Agg')
import geopandas as gpd
import os
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import fiona

def get_gdb_layers(gdbfolderpath):
    layers = fiona.listlayers(gdbfolderpath)
    for lay in layers:
        print(lay)
        
def import_gdb_layer(gdbfolderpath, layer): 
    gdf = gpd.read_file(gdbfolderpath, layer = layer)
    return gdf
            
def filter_gdf_by_gdf(gdf1,gdf2,crs):
    gdf1 = gdf1.to_crs(crs)
    gdf2 = gdf2.to_crs(crs)
    gdf1 = gpd.sjoin(gdf1,gdf2, how = 'inner')
    return gdf1



# try plotting the catchments
folder_catch = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH', 'Cuencas')

catch_file = os.path.join(folder_catch, 'Cuencas_DARH_2015_AOHIA_ZC.geojson')

# folder for the CQ database
folder_CQ = os.path.join('..', 'SIG', 'CQA5868', 'CQA5868_GDB', \
        'Estaciones_DGA.gdb')
gdf_WQ = import_gdb_layer(folder_CQ, 'Promedios')

gdf_catch = gpd.read_file(catch_file)
gdf_catch2 = gdf_catch.copy().to_crs('EPSG:4326')

gdf_WQ = filter_gdf_by_gdf(gdf_WQ, gdf_catch, 'EPSG:32719')

#gdf_WQ.plot()
#plt.show()



# plot the basemap and bubble plot
llcrnrlon = gdf_catch2.geometry.bounds['minx'].min()
llcrnrlat = gdf_catch2.geometry.bounds['miny'].min()
urcrnrlon = gdf_catch2.geometry.bounds['maxx'].max()
urcrnrlat = gdf_catch2.geometry.bounds['maxy'].max()

#----- Basemap
fig = plt.figure()
ax = fig.add_subplot(111)
m = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, \
        urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, \
        resolution='i',projection='cyl', lon_0=-71.090, \
        lat_0 = -35.129, epsg=4326, ax = ax)

m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m.fillcontinents(color='grey',lake_color='aqua', alpha=0.3)
m.drawcoastlines(linewidth=0.1, color="white")

gdf_WQ = gdf_WQ.to_crs('epsg:4326')


col = 'Arsenico_t'
subset = gdf_WQ[gdf_WQ[col]>-999].copy()

m.scatter(x = subset['geometry'].x,
          y = subset['geometry'].y,latlon=True,
          s = subset[col]*1e3,
          cmap = 'viridis')
gdf_catch2.plot(ax=ax, fc='none', ls='--', ec='black')
plt.title('Basemap')
plt.show()

#------ Geopandas
fig = plt.figure()
ax= fig.add_subplot(111)
gdf_catch2.plot(ax=ax, fc='none', ec='black', ls='--')
ax.scatter(x=subset.geometry.x, 
           y=subset.geometry.y,
           s=subset[col]*1000)
plt.show()

