# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 12:56:17 2020

@author: ccalvo
"""

import pyproj
from pyproj import Proj
from pyproj import CRS

# proyección

p = pyproj.Proj(proj='utm', zone="19S", ellps='WGS84')

p = Proj("+proj=utm +zone=19s, +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

x,y = p(35.6697,-71.3433)
print(x)
print(y)


isn2004=pyproj.CRS("+proj=lcc +lat_1=64.25 +lat_2=65.75 +lat_0=65 +lon_0=-19 +x_0=1700000 +y_0=300000 +no_defs +a=6378137 +rf=298.257222101 +to_meter=1") 


wgs84=pyproj.CRS("EPSG:4326")
