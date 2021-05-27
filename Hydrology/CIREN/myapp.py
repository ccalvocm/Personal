#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 13:14:36 2021

@author: faarrosp
"""

# import libraries
import pandas as pd
import geopandas as gpd
import os
from matplotlib import pyplot as plt
import contextily as ctx
import random
from adjustText import adjust_text
from bokeh.tile_providers import CARTODBPOSITRON, get_provider, ESRI_IMAGERY, STAMEN_TERRAIN
import datetime

###### IMPORTANTE #######
'''
Para correr el servidor Bokeh y compartirlo por ngrok, ejecutar desde la
consola (con entorno activado)
bokeh serve myapp.py --allow-websocket-origin=cdea4a37eb69.ngrok.io

https://cdea4a37eb69.ngrok.io

donde c48a3827446e es el address que nos da ngrok
'''

# paths
folder_catchment = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH')
path_catchment = os.path.join(folder_catchment, 'Cuencas', 'Cuencas_DARH_2015_AOHIA_ZC.geojson')
path_subcatchment = os.path.join(folder_catchment, 'Subcuencas', 'SubCuencas_DARH_2015_AOHIA_ZC.geojson')

folder_borehole = os.path.join('..', 'Etapa 1 y 2', 'Aguas subterráneas')
path_borehole = os.path.join(folder_borehole, 'Pozos_DGA_CFA.xlsx')

folder_hidrografia = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Hidrografia')
path_hidrografia = os.path.join(folder_hidrografia, 'RedHidrograficaUTM19S_AOHIA_ZC.geojson')

#-------- create the dataframes/geodataframes
path_pozos_georref = os.path.join(folder_borehole, 'Pozos_DGA_CFA_georref.geojson')
gdf_pozos = gpd.read_file(path_pozos_georref)
gdf_pozos = gdf_pozos.to_crs('EPSG:3857')
gdf_pozos['FECHA'] = pd.to_datetime(gdf_pozos['FECHA'], dayfirst = True)
gdf_pozos['Valor'] = gdf_pozos['Valor'].apply(lambda x: round(x,2))
gdf_pozos_ts = gdf_pozos.copy()
gdf_pozos['YEAR'] = gdf_pozos['FECHA'].dt.year
gdf_pozos.set_index('YEAR', inplace = True)
#gdf_pozos.head()

fields = ['CODIGO', 'NOMBRE', 'COD_CUENCA', 'NOM_CUENCA', 'COD_DGA',
          'NOM_DGA', 'geometry']
dic_pozos = gdf_pozos[fields].drop_duplicates().sort_values(by = 'CODIGO')
dic_pozos.set_index('CODIGO', inplace = True)

gdf_catchment = gpd.read_file(path_catchment)
gdf_catchment = gdf_catchment.to_crs('EPSG:3857')


gdf_pozos_monavg = gdf_pozos.pivot_table(index = 'FECHA',
                                             columns = 'CODIGO',
                                             values = 'Valor')
gdf_pozos_monavg = gdf_pozos_monavg.resample('M').mean()
gdf_pozos_monavg.fillna(method = 'ffill', axis = 0)
#%% Visualizador de pozos

# Perform necessary imports
from bokeh.io import output_file, show, output_notebook
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, GeoJSONDataSource, HoverTool
from bokeh.models import BoxZoomTool, ResetTool, PanTool, WheelZoomTool
from bokeh.plotting import figure

ph = 800
pw = 700

ph2 = 300
pw2 = 700

datestart = pd.to_datetime('1910-10-31')
dateend = pd.to_datetime('2020-10-31')


#-------- Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : dic_pozos.loc[dic_pozos.index, 'geometry'].x,
    'y'       : dic_pozos.loc[dic_pozos.index, 'geometry'].y,
    'Pozo'      : dic_pozos.index.values,
    'Valor' : gdf_pozos_monavg.loc[datestart,dic_pozos.index],
    #'pop'      : (data.loc[1970].population / 20000000) + 2,
    'Nombre'      : dic_pozos.loc[dic_pozos.index,'NOMBRE'],
    'Cuenca' : dic_pozos.loc[dic_pozos.index, 'NOM_CUENCA']
})

#-------- Make the ColumnDataSource: source2 (Pozos timeseries)
source2 = ColumnDataSource(data={
    'x'       : [datestart, dateend],
    'y'       : [0,0]})

#--------

#-------- Min and max values for plot 
# Save the minimum and maximum values of the fertility column: xmin, xmax
xmin = min(gdf_catchment.geometry.bounds['minx'])
xmax = max(gdf_catchment.geometry.bounds['maxx'])
 
ymin = min(gdf_catchment.geometry.bounds['miny'])
ymax = max(gdf_catchment.geometry.bounds['maxy'])
 
#-------- Create the figure: plot
plot = figure(title='Profundidad pozos en 1970', plot_height=ph, plot_width=pw, tools = '',
              x_range=(xmin, xmax), y_range=(ymin, ymax),
              match_aspect = True)
#--------

#-------- Create the figure: plot
plot2 = figure(title='Serie de tiempo Pozo', plot_height=ph2, plot_width=pw2, tools = '',
               x_axis_type = 'datetime')#,
              #x_range=(xmin, xmax), y_range=(ymin, ymax),
              #match_aspect = True)




#-------- Add basemap
tile_provider = get_provider(ESRI_IMAGERY)
plot.add_tile(tile_provider)
#--------

#-------- Add catchment geometries for context
geosource1 = GeoJSONDataSource(geojson = gdf_catchment.to_json())
cuencas = plot.patches('xs', 'ys', source = geosource1,
                    fill_color = None,
                    line_color = 'red',
                    line_width = 2.0,
                    fill_alpha = 1)

#-------- add Hover Tool
hover = HoverTool(tooltips = [('Nombre', '@Nombre'),
                              ('Codigo', '@Pozo'),
                              ('Cuenca', '@Cuenca'),
                              ('Profundidad', '@Valor')])
plot.add_tools(hover)

hover2 = HoverTool(tooltips = [('Fecha', '@x{%F}'),
                               ('Profundidad (m)', '@y{%0.1f}')],
                   formatters = {'@x': 'datetime',
                                 '@y': 'printf'})
plot2.add_tools(hover2)
#-------- 

#-------- add Reset Tool
reset = ResetTool()
plot.add_tools(reset)
#--------

#-------- add Pan Tool
pan = PanTool()
plot.add_tools(pan)

#-------- add WheelZoomTool
zoom = WheelZoomTool()
plot.add_tools(zoom)

#-------- Add circle glyphs to the plot
plot.circle(x='x', y='y', fill_alpha=0.8, source=source)
plot2.circle(x='x', y='y', fill_alpha=0.8, source=source2)
 
# Set the x-axis label
plot.xaxis.axis_label ='Coordenada UTM Este (m)'
plot2.xaxis.axis_label ='Fecha'
 
# Set the y-axis label
plot.yaxis.axis_label = 'Coordenada UTM Norte (m)'
plot2.yaxis.axis_label = 'Nivel freático pozo (m)'

# Set the grid to be invisible
plot.xgrid.visible = False
plot.ygrid.visible = False

plot.add_tools(BoxZoomTool(match_aspect=True))

 
# Add the plot to the current document and add a title
#curdoc().add_root(plot)
curdoc().title = 'Pozos'

#%% 
# # Make a list of the unique values from the region column: regions_list
profundidiades = gdf_pozos.Valor.unique().tolist()
 
# # Import CategoricalColorMapper from bokeh.models and the Spectral6 palette from bokeh.palettes
from bokeh.models import CategoricalColorMapper, LinearColorMapper, ColorBar, ContinuousTicker
from bokeh.palettes import Spectral6, Viridis256
 
# # Make a color mapper: color_mapper
# color_mapper = CategoricalColorMapper(factors=cuencas_list, palette=Spectral6)
 
mapper = LinearColorMapper(palette=Viridis256, low = gdf_pozos['Valor'].min(), high = gdf_pozos['Valor'].max())
colors = {'field': 'Valor', 'transform': mapper}

# # Add the color mapper to the circle glyph
plot.circle(x='x', y='y', fill_alpha=1.0, source=source,
            fill_color= {'field': 'Valor', 'transform': mapper},
            line_color = {'field': 'Valor', 'transform': mapper},
            legend='Pozos DGA')

# -------- Add the colorbar
color_bar_height = ph
color_bar_width = 180

color_bar = ColorBar(color_mapper = mapper, label_standoff = 12, border_line_color = None, location = (0,0))
color_bar_plot = figure(title="Nivel freático pozo (m)", title_location="right", 
                        height=color_bar_height, width=color_bar_width, 
                        toolbar_location=None, min_border=0, 
                        outline_line_color=None)
color_bar_plot.add_layout(color_bar, 'right')
color_bar_plot.title.align = 'center'
color_bar_plot.title.text_font_size = '12pt'
          
# # Set the legend.location attribute of the plot to 'top_right'
# plot.legend.location = 'top_right'
 
# # Add the plot to the current document and add the title
# curdoc().add_root(plot)
# curdoc().title = 'Profundidad pozos 1970'

#%%

# Import the necessary modules
from bokeh.layouts import row, widgetbox, column, layout
from bokeh.models import Slider, Select, DateRangeSlider, DateSlider


dic_COD_CUENCA = {'Todas': ['0600', '1300', '0701', '0703'],
                  'Rio Maipo': ['1300'],
                  'Rio Rapel': ['0600'],
                  'Rio Mataquito': ['0701'],
                  'Rio Maule': ['0703']}
 
# Define the callback function: update_plot
def update_plot(attr, old, new):
    # Set the yr name to slider.value and new_data to source.data
    catch = select_catch.value
    selected = dic_COD_CUENCA[catch]
    sub_gdf_pozos = dic_pozos[dic_pozos['COD_CUENCA'].isin(selected)]
    date = datetime.datetime.fromtimestamp(tslider.value/1000)
    datestring = '-'.join([str(date.day),str(date.month),str(date.year)])
    date = pd.to_datetime(datestring, dayfirst = True)
    plot.title.text = 'Profundidad media mensual Pozos en ' + datestring
    newdate = date.replace(day = 1) - datetime.timedelta(days = 1)
    new_data = {
        'x'       : sub_gdf_pozos.loc[sub_gdf_pozos.index, 'geometry'].x,
        'y'       : sub_gdf_pozos.loc[sub_gdf_pozos.index, 'geometry'].y,
        'Pozo' : sub_gdf_pozos.index.values,
        'Nombre'     : sub_gdf_pozos.loc[sub_gdf_pozos.index, 'NOMBRE'],
        'Cuenca'  : sub_gdf_pozos.loc[sub_gdf_pozos.index, 'NOM_CUENCA'],
        'Valor' : gdf_pozos_monavg.loc[newdate, sub_gdf_pozos.index]
    }
    source.data = new_data
    # Add title to figure: plot.title.text
    #plot.title.text = 'Registro Pozos en %d' % date
def update_select_pozo(attr,old,new):
    catch = select_catch.value
    selected = dic_COD_CUENCA[catch]
    subgrupo_pozos_cuenca = dic_pozos[dic_pozos['COD_CUENCA'].isin(selected)]
    #lista = subgrupo_pozos_cuenca.index + ' ' + subgrupo_pozos_cuenca['NOMBRE']
    lista = subgrupo_pozos_cuenca.index
    select_pozo.options = lista.sort_values().to_list()
    select_pozo.value = lista.sort_values().to_list()[0]
    
def update_plot2(attr, old, new):
    pozo = select_pozo.value
    new_data2 = {
        'x' : gdf_pozos_ts[gdf_pozos_ts['CODIGO'].isin([pozo])]['FECHA'],
        'y' : -gdf_pozos_ts[gdf_pozos_ts['CODIGO'].isin([pozo])]['Valor']
        }
    source2.data = new_data2
 
# # Make a slider object: slider
# slider = Slider(start = 1970, end = 2019, step = 1, value = 1970, title = 'Year')
# Attach the callback to the 'value' property of slider
# slider.on_change('value', update_plot)

#--------- Time Slider
tslider = DateSlider(start = datestart, end = dateend,
                     value = datestart) 
# Attach the callback to the 'value' property of slider
tslider.on_change('value', update_plot)


#--------- Dropdown select
select_catch = Select(
    options = ['Todas', 'Rio Maipo', 'Rio Rapel', 'Rio Mataquito',
               'Rio Maule'],
    value = 'Todas',
    title = 'Cuenca')

#--------- Dropdown select (pozo)
opt_pozos = ['--']
select_pozo = Select(
    options = ['--'],
    title = 'Pozo')
 
select_catch.on_change('value', update_plot, update_select_pozo)
select_pozo.on_change('value', update_plot2)



# Make a row layout of widgetbox(slider) and plot and add it to the current document
#layout = row(widgetbox(select_catch, tslider), plot, color_bar_plot)
#curdoc().add_root(layout)

#row2 = column([plot,plot])
grid = layout([[widgetbox(select_catch, tslider, select_pozo), plot, color_bar_plot],
        [plot2]], sizing_mode='fixed')
curdoc().add_root(grid)


