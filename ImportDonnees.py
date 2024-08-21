import pandas as pd

def ImportDoonnees(chemin_donnees, chemin_MP, chemin_contraintes):
    # df est la base de données qui contient les données des essais de traction et de spectro
    df = pd.read_excel(chemin_donnees)  
    df.drop(columns=['Numéro de four'], inplace=True) 

    # formatter la colonne 'Recette' pour qu'elle soit uniforme
    for i in range(len(df)):
        if 'GS 400-15' in df['Recette'][i] or 'GS 400-18' in df['Recette'][i]:
            df.loc[i,'Recette'] = 'GS 400-15/18'
        elif 'GS 450-10' in df['Recette'][i]:
            df.loc[i,'Recette'] = 'GS 450-10'
        elif 'GS 500-7' in df['Recette'][i]:
            df.loc[i,'Recette'] = 'GS 500-7'
        elif 'GS 600-3' in df['Recette'][i]:
            df.loc[i,'Recette'] = 'GS 600-3'

    # extraction des éléments chimiques de la base de données
    elements_chimiques = df.columns.to_list()
    to_remove =['Recette', 'Conforme', 'Rm', 'Moyenne allongement']
    for elmt in to_remove:
        elements_chimiques.remove(elmt)
    elements_chimiques = set(elements_chimiques) # ensemble des éléments chimiques présents dans les données

    # import de la base des matières premières 
    MP = pd.read_excel(chemin_MP) 
    MP.drop(columns=['Code Article'], inplace=True)
    # filtrage des matières premières métalliques
    metallique = {}
    for i in range(len(MP)):
        metallique[MP['Article'][i]] = MP['Metallique'][i]
    # création d'un dictionnaire pour les composants des matières premières
    MP_compo = MP.copy()
    del MP_compo['Prix']
    del MP_compo['Metallique']
    MP_compo = MP_compo.set_index('Article').T.to_dict()

    # import des contraintes sur les composants
    Compo_contraintes = pd.read_csv(chemin_contraintes, sep=';')
    
    return df, metallique, elements_chimiques, MP_compo, Compo_contraintes, MP

