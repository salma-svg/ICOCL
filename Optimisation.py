from ImportDonnees import ImportDoonnees
from Fonctions import ONO, THIELMANN, MAYER, Impurte, Ferrite, save_errors, adjust_constraints
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, PulpSolverError, COIN_CMD, PULP_CBC_CMD, pulp
import os
import joblib
import warnings
import sys
import tkinter as tk





def optimiser_nuance(nom_nuance, dossier_donnees, faire_prediction):
    # Importation des données
    _, metallique, elements_chimiques, MP_compo, Compo_contraintes, MP = ImportDoonnees(f'{dossier_donnees}/data_nettoyées.xlsx', f'{dossier_donnees}/MP.xlsx', f'{dossier_donnees}/Contraintes_composants.csv')
    # Importation des modèles de prédiction de la résistance mécanique et de l'allongement
    if faire_prediction == 'oui':
        from ModelePrediction import prediction
        rf_model_rm, rf_model_all, moyenne, ecart_type, _, _, _ = prediction(dossier_donnees)
    else:
        try :
            rf_model_rm, rf_model_all, moyenne, ecart_type = joblib.load(f'{dossier_donnees}/modelePrediction/rf_model_rm.pkl'), joblib.load(f'{dossier_donnees}/modelePrediction/rf_model_all.pkl'), joblib.load(f'{dossier_donnees}/modelePrediction/moyenne.pkl'), joblib.load(f'{dossier_donnees}/modelePrediction/ecart_type.pkl')
        except FileNotFoundError:
            erreur = "Les modèles de prédiction n'ont pas été trouvés. Veuillez d'abord les générer"
            save_errors(erreur, dossier_donnees, nom_nuance)
            print(erreur)
            return
    # Définition des colonnes pour les matières premières
    colonne_dispo = f'Disponible_{nom_nuance}'
    colonne_min = f'Part Min_{nom_nuance}'
    colonne_max = f'Part Max_{nom_nuance}'
    seuil_qualite_ONO = 0.6
    seuil_qualite_Thielmann = 2
    seuil_qualite_Mayer = 0.02
    # Détermination des seuils RM et Allongement en fonction de la nuance
    if 'GS 400-15' in nom_nuance or 'GS 400-18' in nom_nuance:
        seuil_rm = 400
        seuil_all = 15
    elif 'GS 450-10' in nom_nuance:
        seuil_rm = 450
        seuil_all = 10
    elif 'GS 500-7' in nom_nuance:
        seuil_rm = 500
        seuil_all = 7
    elif 'GS 600-3' in nom_nuance:
        seuil_rm = 600
        seuil_all = 3

    # Matières premières disponibles MP_dispo
    MP_nuance = MP[['Article', 'Prix', colonne_dispo, colonne_min, colonne_max]]
    MP_nuance.rename(columns={colonne_dispo: 'Disponible', colonne_min: 'Part Min', colonne_max: 'Part Max'}, inplace=True)

    # Filtrage des matières premières disponibles
    MP_nuance = MP_nuance[MP_nuance['Disponible'] == 1]
    MP_dispo = MP_nuance['Article'].to_list()

    # Composition chimique des matières premières disponibles
    MP_compo_nuance = {key: value for key, value in MP_compo.items() if key in MP_dispo}
    # Prix des matières premières disponibles
    prix_MP = MP_nuance.set_index('Article')['Prix'].to_dict()

    # Contraintes de composition pour chaque élément chimique pour la recette
    Compo_contraintes_nuance = Compo_contraintes[Compo_contraintes['Recette'] == nom_nuance]
    Compo_contraintes_nuance = Compo_contraintes_nuance.drop(columns=['Recette']).to_dict(orient='records')[0]

    # Division des contraintes borne inférieure et supérieure pour chaque élément chimique
    borne_inf_compo = {}
    borne_sup_compo = {}

    for key, value in Compo_contraintes_nuance.items():
        component_name = key.split('_')[0]
        if 'min' in key:
            borne_inf_compo[component_name] = value
        elif 'max' in key:
            borne_sup_compo[component_name] = value

    # Compléter les dictionnaires avec les éléments qui n'ont pas de contraintes, on met 0 pour la borne inférieure et 1 pour la borne supérieure
    for elmt in elements_chimiques:
        if elmt not in borne_inf_compo:
            borne_inf_compo[elmt] = 0
        if elmt not in borne_sup_compo:
            borne_sup_compo[elmt] = 1

    # Optimisation

    prob = LpProblem(f"Optimisation d'utilisation de matières premières pour {nom_nuance}", LpMinimize)

    # Création des variables de décision proportion de chaque matière première à utiliser
    variables = LpVariable.dicts("Proportion", MP_dispo, lowBound=0, upBound=1, cat='Continuous')
    # Création des variables binaires, 1 si la matière première est utilisée, 0 sinon
    binary_vars = LpVariable.dicts("Binary", MP_dispo, 0, 1, cat='Binary')

    # Fonction objectif
    prob += lpSum(prix_MP[mp] * variables[mp] for mp in MP_dispo), "Minimiser le coût"

    # Contraintes
    ## Contraintes de borne inférieure et supérieure pour chaque matière première
    for var in MP_dispo:
        part_min = MP_nuance.loc[MP_nuance['Article'] == var, 'Part Min'].values[0]
        part_max = MP_nuance.loc[MP_nuance['Article'] == var, 'Part Max'].values[0]
        if 'Retour' not in var:
            prob += variables[var] >= part_min * binary_vars[var], f"Min_Value_Constraint_{var}"
            prob += variables[var] <= part_max * binary_vars[var], f"Max_Value_Constraint_{var}"
        elif 'Retour' in var:
            prob += variables[var] >= part_min, f"Min_Value_Constraint_{var}"
            prob += variables[var] <= part_max, f"Max_Value_Constraint_{var}"


    ## Contraintes qualité ONO, THIELMANN, MAYER
    prob += (lpSum(ONO(MP_compo_nuance[mp]) * variables[mp] for mp in MP_dispo) <= seuil_qualite_ONO, "Qualité_ONO")
    prob += (lpSum(THIELMANN(MP_compo_nuance[mp]) * variables[mp] for mp in MP_dispo) <= seuil_qualite_Thielmann, "Qualité_Thielmann")
    prob += (lpSum(MAYER(MP_compo_nuance[mp]) * variables[mp] for mp in MP_dispo) <= seuil_qualite_Mayer, "Qualité_Mayer")

    ## Contrainte de proportion, la somme des proportions des matières premières metalliques doit être égale à 1
    prob += lpSum(variables[mp] for mp in MP_dispo if metallique[mp] == 1) == 1
    while True:
        ## Contraintes de chaque élément chimique
        for element in elements_chimiques:
            constraint_min = f"Min_{element}"
            if constraint_min in prob.constraints:
                del prob.constraints[constraint_min]
            constraint_max = f"Max_{element}"
            if constraint_max in prob.constraints:
                del prob.constraints[constraint_max]
            prob += (lpSum(MP_compo_nuance[mp][element] * variables[mp] for mp in MP_dispo) >= borne_inf_compo[element], f"Min_{element}")
            prob += (lpSum(MP_compo_nuance[mp][element] * variables[mp] for mp in MP_dispo) <= borne_sup_compo[element], f"Max_{element}")

        # Résolution du problème
        try :
            prob.solve(PULP_CBC_CMD(msg=False))
        except (PulpSolverError, AttributeError):
            prob.solve(solver=COIN_CMD(path='cbc.exe', msg = False)) # Utilisation du solveur CBC

        # Vérification de l'infaisabilité
        if LpStatus[prob.status] == 'Infeasible':
            # Voir si il y a une solution antérieure qui respecte les contraintes chimiques, si oui, on la retourne, sinon on retourne une erreur
            try :
                if compo_solution :
                    erreur = erreur = "L'algorithme n'a pas pu trouver une solution respectant toutes les contraintes mécaniques.\nLa meilleure solution possible a été retournée."
                    save_errors(erreur, dossier_donnees, nom_nuance)
            except NameError:
                erreur = 'Impossible de satisfaire les contraintes chimiques'
                save_errors(erreur, dossier_donnees, nom_nuance)
            break

        # Calcul de la composition des éléments chimiques optimale trouvée
        compo_solution = {elmt: sum(MP_compo_nuance[mp][elmt] * variables[mp].varValue for mp in MP_dispo) for elmt in elements_chimiques}
        # Mise en forme des résultats
        elements_chimiques = ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Al', 'Cu', 'Ti', 'V', 'Pb', 'Sn', 'Mg', 'As', 'Bi', 'Sb']
        resultats_composition=pd.DataFrame(columns=elements_chimiques)
        for elmt in elements_chimiques:
            resultats_composition[elmt] = [sum(MP_compo_nuance[mp][elmt] * variables[mp].varValue for mp in MP_dispo)]
        for ind in ['ONO', 'Impurte', 'THIELMANN', 'MAYER', 'Ferrite']:
            resultats_composition[ind] = [pulp.value(eval(ind)(compo_solution))]
        resultat_articles = []
        for article in MP_dispo:
            if variables[article].varValue != 0:
                resultat_articles.append({'Article': article, 
                                        'Proportion': variables[article].varValue, 
                                        'Cout': prix_MP[article] * variables[article].varValue})

        # Prédictions et vérification des seuils de resistance mécanique et d'allongement
        compo_list = np.array([compo_solution[elmt] for elmt in elements_chimiques])
        compo_list_normalise = (compo_list - moyenne) / ecart_type
        pred_rm = rf_model_rm.predict(pd.DataFrame([compo_list_normalise], columns=elements_chimiques))[0]
        pred_all = rf_model_all.predict(pd.DataFrame([compo_list_normalise], columns=elements_chimiques))[0]
        if pred_rm < seuil_rm:
            #Les prévisions ne satisfont pas le seuil de la résistance mécanique. Réoptimisation...

            # Ajustement des contraintes pour rf_model_rm
            borne_sup_series = ([pd.Series(borne_sup_compo)[elmt] for elmt in elements_chimiques] - moyenne) / ecart_type
            borne_inf_series = ([pd.Series(borne_inf_compo)[elmt] for elmt in elements_chimiques] - moyenne) / ecart_type
            bounds = list(zip(borne_inf_series, borne_sup_series))
            adjusted_inputs_rm = adjust_constraints(elements_chimiques,rf_model_rm, seuil_rm, compo_list_normalise, bounds)
            adjusted_inputs_rm = adjusted_inputs_rm * ecart_type + moyenne
            for elmt in elements_chimiques:
                if adjusted_inputs_rm[elmt] < borne_sup_compo[elmt]:
                    borne_sup_compo[elmt] = (adjusted_inputs_rm[elmt] + 3 * borne_sup_compo[elmt]) / 4
                elif adjusted_inputs_rm[elmt] > borne_inf_compo[elmt]:
                    borne_inf_compo[elmt] = (adjusted_inputs_rm[elmt] + 3 * borne_inf_compo[elmt]) / 4
            # Réexécution de l'optimisation avec les nouvelles contraintes
            continue
        elif pred_all < seuil_all:
            # Les prévisions ne satisfont pas lee seuil de l'allongement. Réoptimisation...

            # Ajustement des contraintes pour rf_model_all
            borne_sup_series = ([pd.Series(borne_sup_compo)[elmt] for elmt in elements_chimiques] - moyenne) / ecart_type
            borne_inf_series = ([pd.Series(borne_inf_compo)[elmt] for elmt in elements_chimiques] - moyenne) / ecart_type
            bounds = list(zip(borne_inf_series, borne_sup_series))
            adjusted_inputs_all = adjust_constraints(elements_chimiques, rf_model_all, seuil_all, compo_list_normalise, bounds)
            adjusted_inputs_all = adjusted_inputs_all * ecart_type + moyenne
            for elmt in elements_chimiques:
                if adjusted_inputs_all[elmt] < borne_sup_compo[elmt]:
                    borne_sup_compo[elmt] = (adjusted_inputs_all[elmt] + 3 * borne_sup_compo[elmt]) / 4
                elif adjusted_inputs_all[elmt]> borne_inf_compo[elmt]:
                    borne_inf_compo[elmt] = (adjusted_inputs_all[elmt] + 3 * borne_inf_compo[elmt]) / 4
            # Réexécution de l'optimisation avec les nouvelles contraintes
            continue
        break
    
    try :
        resultat_articles = pd.DataFrame(resultat_articles)
        # Exportation des résultats vers le fichier Excel de résultats
        fichier_resultat = f'{dossier_donnees}/resultats_{nom_nuance}.xlsx'
        if fichier_resultat in os.listdir():
            os.remove(fichier_resultat)
        resultat_articles.to_excel(fichier_resultat, index=False)

        # Predictions
        compo_list = {key: compo_solution[key] for key in elements_chimiques}
        compo_list = (list(compo_list.values()) - moyenne) / ecart_type
        compo_df = pd.DataFrame([compo_list], columns=elements_chimiques)
        resultats_composition['Rm'] = rf_model_rm.predict(compo_df)
        resultats_composition['Moyenne allongement'] = rf_model_all.predict(compo_df)
        # Exportation des résultats de composition vers le fichier Excel
        fichier_composition = f'{dossier_donnees}/resultats_composition{nom_nuance}.xlsx'
        if fichier_composition in os.listdir():
            os.remove(fichier_composition)
        resultats_composition.to_excel(fichier_composition, index=False)
        # Sauvegarde des erreurs et infos
        infos = f"Résultats exportés avec succes dans {fichier_resultat} et {fichier_composition}"
        save_errors(infos, dossier_donnees, nom_nuance)

    except (TypeError, ValueError, UnboundLocalError):
        return
    resultats_composition = pd.DataFrame(resultats_composition)
    return resultat_articles, resultats_composition.iloc[:, :-2], resultats_composition.iloc[:, -2:]



if __name__ == '__main__' :
    warnings.filterwarnings("ignore", message="Spaces are not permitted in the name. Converted to '_'")
    pd.options.mode.chained_assignment = None
    # Vérifiez si le dossier des données est fourni en argument
    if len(sys.argv) == 1:
        print("Veuillez fournir le chemin du dossier en argument.")
        sys.exit(1)
    elif len(sys.argv) == 2:
        print("Veuillez indiquer si vous voulez tourner l'algorithme des prédictions ou non.")
        sys.exit(1)
    elif len(sys.argv) == 3:
        print("Veuillez indiquer la fonte à optimiser.")
        sys.exit(1)
    # suppression du fichier d'erreurs s'il existe
    dossier_data = sys.argv[1]
    faire_prediction = sys.argv[2]
    nuance = sys.argv[3]
    fichier_erreurs = os.path.join(dossier_data, 'ErreursEtInfos.txt')
    if os.path.exists(fichier_erreurs):
        os.remove(fichier_erreurs)
    # Optimisation pour chaque nuance
    optimiser_nuance(nuance, dossier_data, faire_prediction)
    print(optimiser_nuance(nuance, dossier_data, faire_prediction))