import os
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

# Charger le modèle
model = load_model('model_unet.h5')  # Assurez-vous que le modèle est dans le même répertoire ou ajustez le chemin

app = Flask(__name__)

# Fonction pour prédire le masque à partir de l'image
def predict_mask(image_bytes):
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

# Fonction pour convertir l'image en base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# Route principale pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Lire l'image directement en mémoire
    image_bytes = file.read()

    # Prédire le masque
    mask = predict_mask(image_bytes)

    # Convertir le mask en image
    mask_image = Image.fromarray(mask.astype(np.uint8))

    # Convertir le mask en base64 pour l'envoi au client
    mask_image_base64 = image_to_base64(mask_image)

    return jsonify({'message': 'Prediction complete', 'mask_image_base64': mask_image_base64})

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
