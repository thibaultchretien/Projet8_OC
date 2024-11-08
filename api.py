import os
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Charger le modèle
model = load_model('model_unet.h5')

app = Flask(__name__)

# Fonction pour prédire le mask à partir de l'image
def predict_mask(image_path):
    # Charger l'image
    image = Image.open(image_path)
    image = image.resize((256, 256))  # Redimensionner l'image à la taille attendue par le modèle
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Ajouter la dimension batch
    image = image / 255.0  # Normaliser les pixels

    # Prédiction
    mask = model.predict(image)
    mask = np.argmax(mask, axis=-1)  # Prendre la classe avec la plus haute probabilité pour chaque pixel
    mask = np.squeeze(mask, axis=0)  # Enlever la dimension batch

    return mask

# Route principale pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Sauvegarder l'image reçue
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Prédire le mask
    mask = predict_mask(file_path)

    # Sauvegarder le mask prédit sous forme d'image
    mask_image = Image.fromarray(mask.astype(np.uint8))  # Convertir le mask en image
    mask_image_path = 'predicted_mask.png'
    mask_image.save(mask_image_path)

    return jsonify({'message': 'Prediction complete', 'mask_image': mask_image_path})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Utiliser le port dynamique de Heroku ou 5000 par défaut
    app.run(host='0.0.0.0', port=port, debug=True)
