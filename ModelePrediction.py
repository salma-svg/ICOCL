import numpy as np
import pandas as pd
from ImportDonnees import ImportDoonnees
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
import joblib
import os

def prediction(dossier_donnees):
    # recupération des données
    data = ImportDoonnees(f'{dossier_donnees}/data_nettoyées.xlsx', f'{dossier_donnees}/MP.xlsx', f'{dossier_donnees}/Contraintes_composants.csv')[0]
    data.dropna(inplace=True)
    data = data.drop(['Conforme', 'Recette'], axis=1)

    # division des données en variables explicatives et valeurs cibles pour faire la prédiction
    # variables explicatives
    X = data.drop(['Rm','Moyenne allongement'], axis=1)
    # valeurs cibles
    y_rm = data['Rm']
    y_allongement = data['Moyenne allongement']

    # Calculer la moyenne et l'écart-type de chaque colonne
    ecart_type = X.std()
    moyenne = X.mean() 
    # Normalisation des données
    X = (X - moyenne) / ecart_type

    # division des données
    X_train, X_test, y_rm_train, y_rm_test, y_allongement_train, y_allongement_test = train_test_split(
        X, y_rm, y_allongement, test_size=0.1, random_state=89
        )
    
    # Création des modèles de prediction
    ## Pour Rm
    rf_model_rm = RandomForestRegressor(n_estimators=50, random_state=87)
    rf_model_rm.fit(X_train, y_rm_train)
    ## Pour l'allongement
    rf_model_all = RandomForestRegressor(n_estimators= 30, random_state=42)
    rf_model_all.fit(X_train, y_allongement_train)

    # Enregistrement des modèles
    try:
        os.makedirs(f'{dossier_donnees}/modelePrediction')
    except FileExistsError:
        pass
    joblib.dump(rf_model_rm, f'{dossier_donnees}/modelePrediction/rf_model_rm.pkl')
    joblib.dump(rf_model_all, f'{dossier_donnees}/modelePrediction/rf_model_all.pkl')
    joblib.dump(moyenne, f'{dossier_donnees}/modelePrediction/moyenne.pkl')
    joblib.dump(ecart_type, f'{dossier_donnees}/modelePrediction/ecart_type.pkl')

    return rf_model_rm, rf_model_all, moyenne, ecart_type, X_test, y_rm_test, y_allongement_test

if __name__ == '__main__':
    rf_model_rm, rf_model_all, _, _, X_test, y_rm_test, y_allongement_test = prediction('data')

    # Faire des prédictions sur l'ensemble de test
    rf_y_pred_rm = rf_model_rm.predict(X_test)
    rf_y_pred_all = rf_model_all.predict(X_test)

    # Calculer les métriques de performance
    rf_r2_rm = r2_score(y_rm_test, rf_y_pred_rm)
    rf_rmse_rm = np.sqrt(mean_squared_error(y_rm_test, rf_y_pred_rm))

    rf_r2_all = r2_score(y_allongement_test, rf_y_pred_all)
    rf_rmse_all = np.sqrt(mean_squared_error(y_allongement_test, rf_y_pred_all))

    print("Pour Rm :")
    print(f"Random Forest R²: {rf_r2_rm:.2f}")
    print(f"Random Forest RMSE: {rf_rmse_rm:.2f}")

    print("\nPour l'allongement :")
    print(f"Random Forest R²: {rf_r2_all:.2f}")
    print(f"Random Forest RMSE: {rf_rmse_all:.2f}")
    print("\n")
