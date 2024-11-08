import os
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Charger le modèle
try:
    print("Chargement du modèle...")
    model = load_model('model_unet.h5')
    print("Modèle chargé.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {str(e)}")

app = Flask(__name__)

# Fonction pour prédire le mask à partir de l'image
def predict_mask(image_path):
    try:
        # Charger l'image
        image = Image.open(image_path)
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

    # Sauvegarder l'image reçue dans le répertoire courant
    file_path = file.filename  # Sauvegarder directement avec le nom du fichier dans le répertoire courant
    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({'error': f"Erreur lors de la sauvegarde de l'image: {str(e)}"}), 500

    # Prédire le mask
    mask, error = predict_mask(file_path)
    if mask is None:
        return jsonify({'error': f"Erreur lors de la prédiction: {error}"}), 500

    # Sauvegarder le mask prédit dans le répertoire courant
    mask_image = Image.fromarray(mask.astype(np.uint8))
    mask_image_path = 'predicted_mask.png'  # Sauvegarder le masque avec un nom fixe dans le répertoire courant
    mask_image.save(mask_image_path)

    return jsonify({'message': 'Prediction complete', 'mask_image': mask_image_path})

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
