# Indicateurs

def ONO(data):
    result = data['Cu'] + data['Ti'] + data['Ni'] + data['Cr'] + data['V'] + data['Al'] + data['As'] + data['Sn'] + data['Pb'] + data['Sb'] + data['Bi']
    return result

def THIELMANN(data):
    result = 4.4 * data['Ti'] + 2 * data['As'] + 2.3 * data['Sn'] + 5 * data['Sb'] + 290 * data['Pb'] +  1.6 * data['Al'] + 370 * data['Bi']
    return result

def MAYER(data):
    result = data['Ti'] + data['Sb'] + data['Pb'] + data['Bi']
    return result

def Impurte(data):
    result = 4.9 * data['Cu'] + 0.37 * (data['Ni'] + data['Cr']) + 7.9 * data['Mo'] + 4.4 * data['Ti'] + 39 * data['Sn'] + 0.44 * data['Mn'] + 5.6 * data['P']
    return result

def Ferrite(data):
    result = 92.3 - 96.2 * data['Mn'] - 211 * data['Cu'] - 14270 * data['Pb'] - 2815 * data['Sb']
    return result

# fonction pour sauvagarder les erreurs

def save_errors(erreurs, dossier_data, recette):
    import os
    fichier_erreurs = os.path.join(dossier_data, 'ErreursEtInfos.txt')

    # Lire le fichier pour vérifier si le message existe déjà
    message = f"Pour la recette {recette}:"
    message_existe_deja = False
    if os.path.exists(fichier_erreurs):
        with open(fichier_erreurs, "r", encoding="utf-8") as f:
            for line in f:
                if message in line:
                    message_existe_deja = True
                    break
    # Écrire le message et les erreurs si le message n'existe pas déjà
    with open(fichier_erreurs, "a+", encoding="utf-8") as f:
        if not message_existe_deja:
            f.write(message + "\n")
        for erreur in erreurs:
            f.write(erreur)
        f.write("\n")
    return

# fonction pour ajuster les contraintes mecanniques en fonction des prédictions
def objective_function(inputs, compo_chimique, model, target):
    import pandas as pd
    import numpy as np
    """ Fonction objectif à minimiser : différence entre la prédiction du modèle et la cible """
    inputs_df = pd.DataFrame([inputs], columns=compo_chimique)
    prediction = model.predict(inputs_df)
    return np.abs(prediction - target)



def adjust_constraints(compo_chimique, model, target, initial_guess, bounds):
    from scipy.optimize import minimize, OptimizeWarning
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        """ Fonction pour ajuster les contraintes en fonction des prédictions """
        result = minimize(objective_function, initial_guess, args=(compo_chimique,model, target),
                            method='L-BFGS-B', tol=1, bounds=bounds)
        for warning in w:
            if issubclass(warning.category, OptimizeWarning):
                if "Initial guess is not within the specified bounds" in str(warning.message):
                    continue  # Ignorer ce warning
                else:
                    print(f"Warning: {warning.message} (in {warning.filename}, line {warning.lineno})")
    adjusted_inputs = result.x
    return adjusted_inputs

if __name__ == '__main__':
    import joblib
    import pandas as pd
    rf_model_all = joblib.load(r'data\rf_model_all.pkl')
    moyenne = joblib.load(r'data\moyenne.pkl')
    ecart_type = joblib.load(r'data\ecart_type.pkl')
    composition = ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Al', 'Cu', 'Ti', 'V', 'Pb', 'Sn', 'Mg', 'As', 'Bi', 'Sb']
    bounds = [(0.5362213674064528, 1.2799521802171145), (-5.762053085801211, -3.9091407422968225), (-3.166325308078445, -0.8766918253381608), (-2.195839638861157, -0.09086546730176737), (-2.025399361136743, 190.75221806213858), (-2.446486341519367, -0.37739573862710013), (-1.2511690934195112, 1.398278242857622), (-2.361584256624182, -0.013087950106225722), (-4.133611868256425, 350.1419454435378), (-1.024644422983117, -0.15511969242644788), (-5.745465399317377, 480.87918667601815), (-1.8413711933196015, 595.7517029695854), (-6.054231303684124, 2984.437715185499), (-2.80385136705183, -0.28683885597619274), (-4.715952855894124, 109.0915585837217), (-1.6787779468227422, 421.7258877382195), (-4.620642969665148, 3599.3757716017894), (-3.0967154590050563, 704.0455820110379)]
    initial_guess = [3.549999985, 2.000000005, 0.2, 0.025219529, 0.029375106, 0.049718045, 0.005462833, 0.007604128, 0.012372521, 0.062023945, 0.007233175, 0.003, 0.001951348, 0.015, 0.016471393, 0.001377852, 0.001620868, 0.005037761]
    initial_guess = (initial_guess - moyenne) / ecart_type
    adj = adjust_constraints(composition, rf_model_all, 20, initial_guess, bounds)
    print(adj, rf_model_all.predict(pd.DataFrame([adj], columns=composition)))
