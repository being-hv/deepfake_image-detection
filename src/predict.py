import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

def load_and_preprocess_image(image_path, img_size=(256, 256)):
    """Loads an image and preprocesses it for model prediction."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img, dtype=np.float32)
    # the dataset prepare doesn't strictly normalize if the backbone does it, 
    # but we will just pass raw pixels to the model assuming it handles preprocessing (keras applications do inside the model if tied correctly)
    return np.expand_dims(img_array, axis=0)

def predict(args):
    print(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    
    print(f"Analyzing Image: {args.image_path}")
    x = load_and_preprocess_image(args.image_path)
    
    prediction = model.predict(x, verbose=0)
    prob_fake = float(prediction[0][0])
    
    label = "FAKE" if prob_fake > 0.5 else "REAL"
    confidence = prob_fake if label == "FAKE" else (1.0 - prob_fake)
    
    print("="*40)
    print(f"Image      : {args.image_path}")
    print(f"Prediction : {label}  (confidence: {confidence*100:.1f}%)")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict single image")
    parser.add_argument('--image_path', type=str, required=True, help='Path to image')
    parser.add_argument('--model_path', type=str, default='models/saved_model/best_model.h5', help='Path to model')
    
    args = parser.parse_args()
    predict(args)
