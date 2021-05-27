# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:18:04 2021

@author: ccalvo
"""



def idea1():
    
    from styleframe import *
    import os
    
    root = r'E:\CIREN'
    root = r'C:\Users\ccalvo'
    folder = root+r'\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\scripts'
    os.chdir(folder)
    
#    sf = StyleFrame.read_excel(folder+'\\test_format.xlsx', read_style=True, use_openpyxl_styles=True)
#    valor_nuevo = ['Modified 1']
#    sf.to_excel('test_format_mod.xlsx', header = False).save()

def idea2():
    import numpy as np
    from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
        Plot, Figure, Matrix, Alignat
    from pylatex import Document, LongTable, MultiColumn, MultiRow
    from pylatex import Document, Section, Subsection, Command
    from pylatex.utils import italic, NoEscape
    from pylatex.utils import italic
    import os
    import json
    
    if __name__ == '__main__':
        latex_document = 'PlantillaFA.tex'
        with open(latex_document) as file:
            tex = file.read()
        
        doc = Document(page_numbers=True)
        doc.append((tex))           
        doc.generate_pdf("Anexo_referencias",compiler='pdfLaTeX')


def loadJSON():
    import json
    f = open('AOHZC_CC.json',encoding = 'utf-8') 
    data = json.load(f) 

def FA():
    from biblib import FileBibDB
    from pybtex.database import parse_file
    import os
    

    bibdata = parse_file('prueba2FAA.bib', bib_format = 'bibtex')
    
    for entry in bibdata.entries:
        title = bibdata.entries[entry].fields['title']
        print(title)
        
    from biblib import FileBibDB
    from pybtex.database import parse_file
    import os
    
    
    
    # path = os.path.join('..',
    #                     'Etapa 1 y 2',
    #                     'Bibliografía',
    #                     'pruebaFAA.bib')
    
    
    # db = FileBibDB('prueba2FAA.bib')
    bibdata = parse_file('prueba2FAA.bib', bib_format = 'bibtex')
    
    
    for entry in bibdata.entries:
        title = bibdata.entries[entry].fields['title'][1:-1]
        print(title)
        
    with open(r'pruebaFAA.tex', 'r') as ogfile:
        text = ogfile.read()
        text_new = text.replace('xxxTituloxxx', title)
        
        with open('output.tex', 'w') as outputfile:
            outputfile.write(text_new)
            
    os.system('pdflatex output.tex')





