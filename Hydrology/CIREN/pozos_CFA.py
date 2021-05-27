import pandas as pd
import os
from matplotlib import pyplot as plt

path = os.path.join('..', 'Etapa 1 y 2', 'Aguas subterráneas', 'Pozos_DGA_CFA.xlsx')

df = pd.read_excel(path, sheet_name = 'BNAT_Niveles_Poz')

print(df.columns)
