import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Charger les nouvelles données
data = pd.read_csv("C:/Users/DELL/Desktop/isetn/pfe/machine-learning/data/FoncierNonBati_data_biens.csv")

# Sélection des features avec le nouveau champ 'type_terrain'
X = data[['lotissement', 'superficie', 'localite', 'type_terrain', 'coutInvestissements']]
y = data['valeurAcquisFCFA']

# Colonnes catégorielles mises à jour
categorical_features = ['lotissement', 'localite', 'type_terrain']  # Ajout de type_terrain

# Pipeline de prétraitement mis à jour
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Pipeline complet (identique mais avec nouvelles features)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning (peut être ajusté)
param_grid = {
    'regressor__n_estimators': [100, 150, 200],  # Augmenté pour plus de complexité
    'regressor__max_depth': [None, 15, 25],      # Profondeurs ajustées
    'regressor__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)  # cv augmenté à 5
grid_search.fit(X_train, y_train)

# Évaluation
print("Meilleurs paramètres:", grid_search.best_params_)
y_pred = grid_search.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Erreur quadratique moyenne : {mse}")
print(f"R² : {r2}")

# Sauvegarde du nouveau modèle
joblib.dump(grid_search.best_estimator_, 'C:/Users/DELL/Desktop/isetn/pfe/machine-learning/models/model_fonciernonbati.pkl')
print("Nouveau modèle sauvegardé dans 'model_fonciernonbati_v2.pkl'")