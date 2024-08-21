import numpy as np
import pandas as pd
from Fonctions import extract_and_format, replace_format, extract_and_format_traction

data_traction = pd.read_excel('data/Essai_Traction.xlsx') # données de traction
data_spectro = pd.read_excel('data/Spectro.xlsx') # données de spectro
data_spectro = data_spectro.dropna() 
data_spectro = data_spectro[data_spectro['Base'] == 'reste'] # On ne garde que les mesures avec le four
data_spectro.columns = data_spectro.columns.str.replace(' ', '')
data_spectro['numéropoche'] = data_spectro['numéropoche'].fillna('')

# Filtrage des données
data_spectro['numéropoche'] = data_spectro['numéropoche'].astype(str)
data_spectro= data_spectro[data_spectro['numéropoche'].str.contains('NF')]
data_spectro= data_spectro[data_spectro['IDNuance'].str.contains('FGS')]
data_spectro['numerofour'] = data_spectro['numéropoche'].str[:7]

columns_to_average = ['C','Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Al', 'Co',
                    'Cu', 'Nb', 'Ti','V', 'W', 'Pb', 'Sn', 'Mg', 'As', 'Zr',
                    'Bi', 'Ce', 'Sb', 'Se', 'Te', 'B', 'Zn', 'La', 'N', 'Fe',
                    'Ceq', 'RS/Mn']

result = data_spectro.groupby(['numerofour', 'IDNuance'],)[columns_to_average].mean().reset_index()

data_spectro_2 = data_spectro.merge(result, on=['numerofour', 'IDNuance'], how='left')
data_spectro_2= data_spectro_2[['IDNuance','numerofour', 'C_y', 'Si_y', 'Mn_y',
       'P_y', 'S_y', 'Cr_y', 'Mo_y', 'Ni_y', 'Al_y', 'Co_y', 'Cu_y', 'Nb_y',
       'Ti_y', 'V_y', 'W_y', 'Pb_y', 'Sn_y', 'Mg_y', 'As_y', 'Zr_y', 'Bi_y',
       'Ce_y', 'Sb_y', 'Se_y', 'Te_y', 'B_y', 'Zn_y', 'La_y', 'N_y', 'Fe_y',
       'Ceq_y', 'RS/Mn_y']]
data_spectro_2.columns = data_spectro_2.columns.str.replace('_y', '')
data_spectro_2.drop_duplicates(subset=['numerofour', 'IDNuance'], inplace=True)
data_spectro_2['numerofour'] = data_spectro_2['numerofour'].str.split('P').str[0]

data_spectro_2['IDNuance'] = data_spectro_2['IDNuance'].apply(extract_and_format)

data_traction = data_traction[['Matière', 'Poche n° / N° de Pièce', 'C / NC', 'RM','Moyenne Allongement %']]

data_traction.rename(columns={
    'Matière': 'IDNuance',
    'Poche n° / N° de Pièce' : 'numerofour',
    'C / NC': 'conforme',
    'Moyenne Allongement %' : 'Allongement'},
    inplace=True)
data_traction.loc[:, 'numerofour'] = data_traction['numerofour'].apply(replace_format)
data_traction.loc[:, 'IDNuance'] = data_traction['IDNuance'].apply(extract_and_format_traction)

columns_to_average = ['RM', 'Allongement']
data_traction[columns_to_average] = data_traction[columns_to_average].apply(pd.to_numeric, errors='coerce')
result_traction = data_traction.groupby(['numerofour', 'IDNuance'])[columns_to_average].mean().reset_index()

# merge des deux dataframes
data = data_spectro_2.merge(data_traction, on=['numerofour', 'IDNuance'], how='inner')
data.loc[data['conforme'] == 'C', 'conforme'] = 1
data.loc[data['conforme'] == 'NC', 'conforme'] = 0
data.loc[(data['IDNuance'] == 'GS 400-15') | (data['IDNuance'] == 'GS 400-18'), 'IDNuance'] = 'GS 400-15/18'
data.rename(columns={'IDNuance': 'Recette', 'RM' : 'Rm', 'Allongement' : 'Moyenne allongement', 'conforme' : 'Conforme'}, inplace=True)

data_2 = data.copy()
data_2.rename(columns={'numerofour': 'Numéro de four'}, inplace=True)
data_2 = data_2[['Numéro de four','Recette', 'C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Al', 'Co', 'Cu', 'Nb', 'Ti', 'V', 'W', 'Pb', 'Sn', 'Mg', 'As', 'Zr', 'Bi', 'Ce', 'Sb', 'Se', 'Te', 'B', 'Zn', 'La', 'N', 'Fe', 'Ceq', 'RS/Mn', 'Rm', 'Moyenne allongement', 'Conforme']]

df = pd.read_excel('data/Traction.xlsx')
df = df.rename(columns=lambda x: x.replace(' [%]', '').replace(' [MPA]', '').replace(' ?', ''))
df.drop_duplicates
#supprimer les colonnes inutiles dans la regression
df.drop(['Impureté', 'Ferrite', 'Purete ONO', 'Purete THIELMANN', 'Purete MAYER'], axis=1, inplace=True)
# supression des lignes avec Rm = 0, ce sont des valeurs aberrantes
df = df[df['Rm'] != 0]
df= df[df['Numéro de four'].str.contains('NF')]
df['Numéro de four'] = df['Numéro de four'].str[:7]
columns_to_average_2 = ['Rm', 'Moyenne allongement', 'C', 'Si', 'Mn',
       'Cu', 'Cr', 'P', 'Ni', 'Mo', 'Sn', 'Sb', 'Al', 'S', 'Mg', 'Pb', 'Ti',
       'As', 'Bi', 'V']
df = df.dropna(subset=columns_to_average_2)
df = df.groupby(['Numéro de four','Recette', 'Conforme'])[columns_to_average_2].mean().reset_index()
df =df.drop_duplicates(
    subset=['C', 'Si', 'Mn', 'Cu', 'Cr', 'P', 'Ni', 'Mo', 'Sn', 'Sb', 'Al', 'S', 'Mg', 'Pb', 'Ti','As', 'Bi', 'V'],
    keep='first')

df.loc[(df['Recette'] == 'GS 400-15') | (df['Recette'] == 'GS 400-18'), 'Recette'] = 'GS 400-15/18'

# on concatene les deux dataframes en ne gardant que les colonnes communes
data_3 = pd.concat([data_2, df], join='inner')
data_3.drop_duplicates(inplace=True)
data_3.to_excel('data/data_nettoyées.xlsx', index=False)