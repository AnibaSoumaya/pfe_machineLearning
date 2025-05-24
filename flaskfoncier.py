from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Charger le modèle sauvegardé
model = joblib.load('C:/Users/DELL/Desktop/isetn/pfe/machine-learning/models/modele_immobilier_niger_optimise.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données de la requête
        data = request.get_json()
        
        # Valider les données d'entrée
        required_fields = ['superficie', 'nature', 'localisation', 'anneeConstruction', 'nbrChambres', 'etatGeneral']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Le champ {field} est manquant'}), 400
        
        # Préparer les données pour la prédiction
        villes_principales = ['Niamey', 'Maradi', 'Zinder', 'Agadez']
        
        input_data = pd.DataFrame({
            'superficie': [float(data['superficie'])],
            'nature': [data['nature']],
            'localisation': [data['localisation'] if data['localisation'] in villes_principales else 'Autre'],
            'age': [2023 - int(data['anneeConstruction'])],
            'nbrChambres': [int(data['nbrChambres'])],
            'prix_m2': [0],  # Valeur factice comme dans votre fonction originale
            'etatGeneral': [data['etatGeneral']]
        })
        
        # Faire la prédiction
        log_prix = model.predict(input_data)
        prix = np.expm1(log_prix)[0]
        
        # Retourner la réponse
        return jsonify({
            'prediction': round(float(prix), 2),
            'currency': 'FCFA',
            'features': data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

""" 
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Charger le modèle sauvegardé
model = joblib.load('C:/Users/DELL/Desktop/isetn/pfe/machine-learning/models/modele_immobilier_niger_optimise.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données de la requête
        data = request.get_json()
        
        # Valider les données d'entrée
        required_fields = ['superficie', 'nature', 'localisation', 'anneeConstruction', 'nbrChambres', 'etatGeneral']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Le champ {field} est manquant'}), 400
        
        # Préparer les données pour la prédiction
        villes_principales = ['Niamey', 'Maradi', 'Zinder', 'Agadez']
        current_year = 2023
        
        input_data = pd.DataFrame({
            'superficie': [float(data['superficie'])],
            'nature': [data['nature']],
            'localisation': [data['localisation'] if data['localisation'] in villes_principales else 'Autre'],
            'age': [current_year - int(data['anneeConstruction'])],
            'nbrChambres': [int(data['nbrChambres'])],  # Correction de l'orthographe ici
            'prix_m2': [0],  # Valeur factice
            'etatGeneral': [data['etatGeneral']]
        })
        
        # Faire la prédiction
        log_prix = model.predict(input_data)
        prix = np.expm1(log_prix)[0]
        
        # Appliquer des ajustements métier si nécessaire
        if data['nature'] == 'Villa' and data['localisation'] == 'Maradi':
            prix *= 1.15
        elif int(data['anneeConstruction']) < 2015:
            prix *= 1.10
            
        # Formater la réponse
        response = {
            'prediction': round(float(prix), 2),
            'currency': 'FCFA',
            'features': data,
            'model_version': '1.0'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e), 'details': 'Vérifiez le format des données envoyées'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) """