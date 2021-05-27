

import os
import geopandas
import pandas

folder = os.path.join('..', 'Etapa 1 y 2', 'Aguas subterráneas')
filename = 'Pozos_DGA_CFA.xlsx'
fp = os.path.join(folder, filename)

dataframe = pandas.read_excel(fp, sheet_name = 'BNAT_Niveles_Poz')

pozos = dataframe.drop_duplicates(subset='CODIGO')

geodataframe = geopandas.GeoDataFrame(pozos, crs='EPSG:32719', geometry = geopandas.points_from_xy(pozos['UTM ESTE'],pozos['UTM NORTE']))

print(geodataframe.head())

geodataframe.to_file('pozos_FAA.shp', driver = 'ESRI Shapefile')

#print(pozos.head())
#print(pozos.columns)


