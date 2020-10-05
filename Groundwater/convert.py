# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:38:52 2020

@author: fcidm
"""

import IPython.nbformat.current as nbf
nb = nbf.read(open('Plot_hds.py', 'r'), 'py')
nbf.write(nb, open('Plot_hds.ipynb', 'w'), 'ipynb')