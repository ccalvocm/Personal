# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 14:55:54 2022

@author: Carlos
"""

import ee
import folium

ee.Authenticate()
ee.Initialize()

Countries=ee.FeatureCollection('users/midekisa/Countries')
Zambia=Countries.filter(ee.Filter.eq('Country','Zambia'))
dataset=ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA').filterDate('2017-01-01',
'2017-12-31').filterBounds((Zambia))
eeimage=dataset.first()

#mapa interactivo