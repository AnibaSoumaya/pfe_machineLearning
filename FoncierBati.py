# ============================================================
# IMPORTATIONS REQUISES
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib

# ============================================================
# 1. CHARGEMENT ET NETTOYAGE DES DONNÉES
# ============================================================
def load_and_clean_data():
    """Charge et nettoie les données immobilières"""
    data = pd.read_csv(
        "C:/Users/DELL/Desktop/isetn/pfe/machine-learning/data/FoncierBati_data_biens.csv"
    )
    print(data.head())

    numeric_cols = ["superficie", "anneeConstruction", "coutAcquisitionFCFA"]
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Valeurs manquantes obligatoires
    data = data.dropna(subset=numeric_cols + ["nature", "localisation"])

    # Filtrage des outliers simples
    data = data[
        (data["coutAcquisitionFCFA"] > 1e6) & (data["coutAcquisitionFCFA"] < 5e8)
    ]
    data = data[(data["superficie"] > 20) & (data["superficie"] < 1000)]

    return data


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
def engineer_features(data):
    """Crée des features pertinentes pour le marché nigérien"""
    data["age"] = 2023 - data["anneeConstruction"]
    data["prix_m2"] = data["coutAcquisitionFCFA"] / data["superficie"]

    # Suppression des biens sans état général
    data = data.dropna(subset=["etatGeneral"])

    # Regroupe les localisations secondaires sous 'Autre'
    grandes_villes = ["Niamey", "Maradi", "Zinder", "Agadez"]
    data["localisation"] = np.where(
        data["localisation"].isin(grandes_villes), data["localisation"], "Autre"
    )

    return data


# ============================================================
# 3. PRÉPARATION DES DONNÉES
# ============================================================
def prepare_data(data):
    """Sépare X et y puis crée les jeux train / test"""
    features = [
        "superficie",
        "nature",
        "localisation",
        "age",
        "nbrChambres",
        "prix_m2",
        "etatGeneral",
    ]
    X = data[features]
    y = np.log1p(data["coutAcquisitionFCFA"])  # log1p pour stabiliser la variance
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ============================================================
# 4. PRÉ-TRAITEMENT COMMUN
# ============================================================
numeric_features = ["superficie", "age", "nbrChambres", "prix_m2"]
categorical_features = ["nature", "localisation", "etatGeneral"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
            ),
            numeric_features,
        ),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ============================================================
# 5. CONSTRUCTION DES MODÈLES
# ============================================================
def build_random_forest():
    """Pipeline Random Forest"""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=18,
                    min_samples_leaf=3,
                    max_features=0.7,
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_linear_regression():
    """Pipeline Régression linéaire (baseline)"""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


# ============================================================
# 6. ÉVALUATION
# ============================================================
def evaluate_model(model, X_test, y_test, label="Modèle"):
    """Évalue et affiche les performances"""
    y_pred = model.predict(X_test)

    # Retour espace valeur originale
    y_test_exp = np.expm1(y_test)
    y_pred_exp = np.expm1(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred_exp))
    r2 = r2_score(y_test_exp, y_pred_exp)
    mape = np.mean(np.abs(y_test_exp - y_pred_exp) / y_test_exp) * 100

    print(f"\n===== PERFORMANCES {label} =====")
    print(f"RMSE  : {rmse:,.2f} FCFA")
    print(f"R²    : {r2:.3f}")
    print(f"MAPE  : {mape:.1f}%")

    return rmse, r2, mape


# ============================================================
# 7. EXÉCUTION PRINCIPALE
# ============================================================
if __name__ == "__main__":
    # -- Chargement et préparation
    data = engineer_features(load_and_clean_data())
    X_train, X_test, y_train, y_test = prepare_data(data)

    # -- RANDOM FOREST
    rf_model = build_random_forest()
    print("Entraînement Random Forest...")
    rf_model.fit(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, label="Random Forest")

    # -- RÉGRESSION LINÉAIRE
    lr_model = build_linear_regression()
    print("\nEntraînement Régression linéaire...")
    lr_model.fit(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test, label="Régression linéaire")

    # -- Sauvegarde du meilleur modèle (Random Forest ici)
    joblib.dump(
        rf_model,
        "C:/Users/DELL/Desktop/isetn/pfe/machine-learning/models/modele_immobilier_niger_optimise.pkl",
    )
    print("\nModèle Random Forest sauvegardé avec succès.")

    # -- Exemples de prédiction avec le meilleur modèle
    exemples = [
        (120, "Maison", "Niamey", 2015, 3, "bon"),
        (200, "Villa", "Maradi", 2018, 4, "neuf"),
        (80, "Maison", "Tahoua", 2010, 2, "moyen"),
    ]

    print("\nEXEMPLES DE PRÉDICTIONS (Random Forest) :")
    for superficie, nature, localisation, annee, chambres, etat in exemples:
        input_df = pd.DataFrame(
            {
                "superficie": [superficie],
                "nature": [nature],
                "localisation": [localisation if localisation in ["Niamey", "Maradi", "Zinder", "Agadez"] else "Autre"],
                "age": [2023 - annee],
                "nbrChambres": [chambres],
                "prix_m2": [0],  # placeholder
                "etatGeneral": [etat],
            }
        )
        log_prix = rf_model.predict(input_df)
        prix = np.expm1(log_prix)[0]
        print(f"- {nature} {superficie} m² à {localisation} : {prix:,.0f} FCFA")
