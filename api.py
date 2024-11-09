import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64
import os

# Charger le modèle de segmentation
model = load_model('model_simple.h5')
#model = load_model('model_unet.h5')

app = Flask(__name__)

# Fonction pour prédire le masque à partir de l'image
def predict_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).resize((256, 256))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)  # Préparation de l'image
    mask = model.predict(image_array)  # Prédiction du masque
    mask = np.argmax(mask, axis=-1).squeeze()  # Classes prédictives et suppression de la dimension batch
    return Image.fromarray(mask.astype(np.uint8))  # Retourne le masque en tant qu'image

# Fonction pour convertir l'image en base64
def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Route principale pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file or file.filename == '':
        return jsonify({'error': 'No image file provided'}), 400

    mask_image = predict_mask(file.read())  # Prédire le masque
    mask_image_base64 = image_to_base64(mask_image)  # Encoder en base64

    return jsonify({'message': 'Prediction complete', 'mask_image_base64': mask_image_base64})

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))

