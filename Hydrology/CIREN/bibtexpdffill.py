#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:51:28 2021

@author: faarrosp
"""

from biblib import FileBibDB
from pybtex.database import parse_file
import os
import re

def replace(string, substitutions):
    substrings = sorted(substitutions, key=len, reverse=True)
    regex = re.compile('|'.join(map(re.escape, substrings)))
    return regex.sub(lambda match: substitutions[match.group(0)], string)


bibdata = parse_file('all.bib', bib_format = 'bibtex')
with open('beg.tex','r') as begdoc:
    begtex = begdoc.read()
    
with open('end.tex', 'r') as enddoc:
    endtex = enddoc.read()

with open('output2.tex', 'w') as outputfile:
    outputfile.write(begtex)
    for entry in bibdata.entries:
        title = bibdata.entries[entry].fields['title'][1:-1]
        year = bibdata.entries[entry].fields['year']
        autores = bibdata.entries[entry].persons['author']
        subs = {'xxxTituloxxx': title,
                'xxxAnoxxx': year}
        # print(title)
        # print(year)
    
        with open('pruebaFAA.tex', 'r') as ogfile:
            text = ogfile.read()
            
            text_new = replace(text, subs)
    
    
        outputfile.write(text_new)
    outputfile.write(endtex)
os.system('pdflatex output2.tex')