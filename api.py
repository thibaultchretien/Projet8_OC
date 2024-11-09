import io
import base64
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image

# Initialize Flask app and load model
app = Flask(__name__)
model = load_model("model_segmentation.h5")

def preprocess_image(image, target_size=(256, 256)):
    """Resize and normalize the input image for the model."""
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

def postprocess_mask(mask):
    """Post-process the model output for visualization."""
    mask = np.squeeze(mask)  # Remove batch dimension
    mask = (mask > 0.5).astype(np.uint8) * 255  # Binarize and scale to 0-255
    return mask

@app.route('/')
def home():
    """Home route to confirm the API is running."""
    return "Welcome to the Image Segmentation API! The API is up and running."

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    input_data = preprocess_image(image)

    # Model prediction
    prediction = model.predict(input_data)
    mask = postprocess_mask(prediction[0])

    # Convert mask to base64 for response
    mask_img = Image.fromarray(mask)
    buffered = io.BytesIO()
    mask_img.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({"predicted_mask": mask_base64})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
