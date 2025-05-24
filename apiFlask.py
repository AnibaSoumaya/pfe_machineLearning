from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

@app.route('/predict/<entity>', methods=['POST'])
def predict(entity):
    try:
        # Construire le chemin du modèle
        model_path = f'models/model_{entity.lower()}.pkl'
        if not os.path.exists(model_path):
            return jsonify({'error': f"Modèle pour l'entité '{entity}' introuvable."}), 404

        # Charger le modèle
        model = joblib.load(model_path)

        # Récupérer les données du JSON envoyé
        input_json = request.get_json()
        input_data = pd.DataFrame([input_json])

        # Faire la prédiction
        prediction = model.predict(input_data)

        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
