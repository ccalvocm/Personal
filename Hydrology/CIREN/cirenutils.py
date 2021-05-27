# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:45:26 2020

@author: farrospide
"""

from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

logociren = './logos/logo_ciren.jpg'
logominagri = './logos/logo_minagri.png'


imgciren = mpimg.imread(logociren)
imgminagri = mpimg.imread(logominagri)

# fig = plt.figure(figsize = (22, 17), tight_layout=True)
def vinetaCIREN(fig):
    # gs = gridspec.GridSpec(3, 2, height_ratios= [1,1,1], width_ratios=[2,1])
    
    # ax1 = fig.add_subplot(gs[:, 0])
    ax1 = plt.subplot(121)
    # ax1.plot(np.arange(0, 1e6, 1000))
    # ax1.set_ylabel('YLabel0')
    # ax1.set_xlabel('XLabel0')
    
    
    # ax2 = fig.add_subplot(gs[0, 1])
    ax2 = plt.subplot(322)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    text = 'SIMBOLOGIA'
    ax2.text(0.5,0.9, text, ha = 'center', fontweight = 'bold')
    
    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    
    # ax3 = fig.add_subplot(gs[1, 1])
    # ax3.axes.xaxis.set_visible(False)
    # ax3.axes.yaxis.set_visible(False)
    
    # vineta datos cartograficos
    # ax = fig.add_subplot(gs[1, 1])
    ax4 = plt.subplot(324)
    
    texto = '\n'.join(['Datos Cartográficos',
                       'Proyección Universal Transversal de Mercator',
                       'Huso: 19 Sur',
                       'Origen de las abcisas:',
                       'Origen de las ordenadas:',
                       'Factor de Escala Meridiano Central',
                       'Datos Geodésicos',
                       'Elipsoide y Datum de Referencia: WGS84'])
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    ax4.text(0.025,0.025, texto, ha = 'left')
    ax4.axes.xaxis.set_visible(False)
    ax4.axes.yaxis.set_visible(False)
    # ax4.axis('off')
    
    
    # vineta CIREN
    # ax5 = fig.add_subplot(gs[2, 1])
    ax5 = plt.subplot(326)
    
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    
    arr_minagri = mpimg.imread(logominagri)
    arr_ciren = mpimg.imread(logociren)
    
    imagebox1 = OffsetImage(arr_minagri, zoom=0.1)
    imagebox2 = OffsetImage(arr_ciren, zoom=0.05)
    
    ab1 = AnnotationBbox(imagebox1, (0.25, 0.6), frameon = False)
    ab2 = AnnotationBbox(imagebox2, (0.65, 0.6), frameon = False)
    
    ax5.add_artist(ab1)
    ax5.add_artist(ab2)
    
    texto = 'Analisis de oferta hidrica y su impacto en la agricultura' \
        + '\n' + 'Zona Centro'
    
    ax5.text(0.5,0.025, texto, ha = 'center')
    # ax5.axis('off')
    ax5.axes.xaxis.set_visible(False)
    ax5.axes.yaxis.set_visible(False)
    
    return ax1,ax2,ax4,ax5

# plt.grid()

# plt.draw()
# # plt.savefig('add_picture_matplotlib_figure.png',bbox_inches='tight')
# plt.show()
