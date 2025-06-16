import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Charger les données
data = pd.read_csv('C:/Users/DELL/Desktop/isetn/pfe/machine-learning/data/carData.csv')

# Vérifier et supprimer les lignes avec valeurs manquantes dans la cible
data = data.dropna(subset=['Selling_Price'])
print(data.head())# Définir X et y
X = data[['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Transmission']]
y = data['Selling_Price']

# Colonnes catégorielles
categorical_features = ['Fuel_Type', 'Transmission']

# 3. PRÉPARATION DES DONNÉES
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GridSearchCV
param_grid = {
    'regressor__n_estimators': [50, 100],
    'regressor__max_depth': [None, 10],
    'regressor__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Résultats
print("Meilleurs paramètres:", grid_search.best_params_)
y_pred = grid_search.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Sauvegarder le modèle
joblib.dump(grid_search.best_estimator_, 'C:/Users/DELL/Desktop/isetn/pfe/machine-learning/models/model_vehicule.pkl')
