# IMPORTATIONS REQUISES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.impute import SimpleImputer

# 1. CHARGEMENT ET NETTOYAGE DES DONNÉES
def load_and_clean_data():
    """Charge et nettoie les données immobilières"""
    data = pd.read_csv('C:/Users/DELL/Desktop/isetn/pfe/machine-learning/data/FoncierBati_data_biens.csv')
    
    numeric_cols = ['superficie', 'anneeConstruction', 'coutAcquisitionFCFA']
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    data = data.dropna(subset=numeric_cols + ['nature', 'localisation'])
    data = data[(data['coutAcquisitionFCFA'] > 1e6) & (data['coutAcquisitionFCFA'] < 5e8)]
    data = data[(data['superficie'] > 20) & (data['superficie'] < 1000)]
    
    return data

# 2. FEATURE ENGINEERING
def engineer_features(data):
    """Crée des features pertinentes pour le marché nigérien"""
    data['age'] = 2023 - data['anneeConstruction']
    data['prix_m2'] = data['coutAcquisitionFCFA'] / data['superficie']
    data = data.dropna(subset=['etatGeneral'])

    villes_principales = ['Niamey', 'Maradi', 'Zinder', 'Agadez']
    data['localisation'] = np.where(
        data['localisation'].isin(villes_principales),
        data['localisation'],
        'Autre'
    )
    
    return data

# 3. PRÉPARATION DES DONNÉES
def prepare_data(data):
    """Prépare les données pour l'entraînement"""
    features = ['superficie', 'nature', 'localisation', 'age', 'nbrChambres', 'prix_m2', 'etatGeneral']
    X = data[features]
    y = np.log1p(data['coutAcquisitionFCFA'])
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 4. CONSTRUCTION DU MODÈLE
def build_model():
    """Construit le pipeline de modélisation"""
    numeric_features = ['superficie', 'age', 'nbrChambres', 'prix_m2']
    categorical_features = ['nature', 'localisation', 'etatGeneral']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=3,
            max_features=0.7,
            bootstrap=True,
            random_state=42,
            n_jobs=-1))
    ])

# 5. ÉVALUATION
def evaluate_model(model, X_test, y_test):
    """Évalue et visualise les performances du modèle"""
    y_pred = model.predict(X_test)
    y_test_exp = np.expm1(y_test)
    y_pred_exp = np.expm1(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred_exp))
    r2 = r2_score(y_test_exp, y_pred_exp)
    mape = np.mean(np.abs(y_test_exp - y_pred_exp) / y_test_exp) * 100
    
    print("\nPERFORMANCE DU MODÈLE")
    print("---------------------")
    print(f"RMSE: {rmse:,.2f} FCFA")
    print(f"R²: {r2:.3f}")
    print(f"Erreur relative moyenne (MAPE): {mape:.1f}%")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_exp, y_pred_exp, alpha=0.5, label='Prédictions')
    plt.plot([y_test_exp.min(), y_test_exp.max()], 
             [y_test_exp.min(), y_test_exp.max()], 'r--', label='Parfaite prédiction')
    plt.xlabel('Prix Réel (FCFA)')
    plt.ylabel('Prix Prédit (FCFA)')
    plt.title('Performance du Modèle Immobilier Niger')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return rmse, r2, mape

# 6. FONCTION DE PRÉDICTION
def predict_price(model, superficie, nature, localisation, annee_construction, nbr_chambres, etat_general):
    """Prédit le prix d'un bien immobilier"""
    villes_principales = ['Niamey', 'Maradi', 'Zinder', 'Agadez']
    
    input_data = pd.DataFrame({
        'superficie': [superficie],
        'nature': [nature],
        'localisation': [localisation if localisation in villes_principales else 'Autre'],
        'age': [2023 - annee_construction],
        'nbrChambres': [nbr_chambres],
        'prix_m2': [0],  # Valeur factice
        'etatGeneral': [etat_general]
    })
    
    log_prix = model.predict(input_data)
    return np.expm1(log_prix)[0]

# EXÉCUTION PRINCIPALE
if __name__ == "__main__":
    data = load_and_clean_data()
    data = engineer_features(data)
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    model = build_model()
    print("Entraînement du modèle en cours...")
    model.fit(X_train, y_train)
    
    rmse, r2, mape = evaluate_model(model, X_test, y_test)
    
    joblib.dump(model, 'C:/Users/DELL/Desktop/isetn/pfe/machine-learning/models/modele_immobilier_niger_optimise.pkl')
    print("\nModèle sauvegardé avec succès sous 'modele_immobilier_niger_optimise.pkl'")
    
    exemples = [
        (120, 'Maison', 'Niamey', 2015, 3, 'bon'),
        (200, 'Villa', 'Maradi', 2018, 4, 'neuf'),
        (80, 'Maison', 'Tahoua', 2010, 2, 'moyen')
    ]
    
    print("\nEXEMPLES DE PRÉDICTIONS:")
    for superficie, nature, localisation, annee, chambres, etat in exemples:
        prix = predict_price(model, superficie, nature, localisation, annee, chambres, etat)
        print(f"- {nature} {superficie}m² à {localisation} ({annee}, {chambres} ch., état: {etat}): {prix:,.0f} FCFA")
