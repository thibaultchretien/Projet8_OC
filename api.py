import os
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

# Charger le modèle
try:
    print("Chargement du modèle...")
    model = load_model('model_unet.h5')
    print("Modèle chargé.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {str(e)}")

app = Flask(__name__)

# Fonction pour prédire le mask à partir de l'image
def predict_mask(image_bytes):
    try:
        # Convertir les bytes en image
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((256, 256))  # Redimensionner l'image
        image = np.array(image)
        image = np.expand_dims(image, axis=0)  # Ajouter la dimension batch
        image = image / 255.0  # Normalisation

        # Prédiction
        mask = model.predict(image)
        mask = np.argmax(mask, axis=-1)  # Prendre la classe la plus probable
        mask = np.squeeze(mask, axis=0)  # Enlever la dimension batch
        return mask
    except Exception as e:
        return None, str(e)

# Route principale pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Lire l'image directement en mémoire
        image_bytes = file.read()

        # Prédire le mask
        mask, error = predict_mask(image_bytes)
        if mask is None:
            return jsonify({'error': f"Erreur lors de la prédiction: {error}"}), 500

        # Sauvegarder le masque prédit dans un format de retour pour l'affichage (sans fichier)
        mask_image = Image.fromarray(mask.astype(np.uint8))
        mask_image_path = 'predicted_mask.png'  # Masque retourné dans un format d'image
        mask_image.save(mask_image_path)

        return jsonify({'message': 'Prediction complete', 'mask_image': mask_image_path})

    except Exception as e:
        return jsonify({'error': f"Erreur: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
