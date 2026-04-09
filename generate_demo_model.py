import os
import tensorflow as tf

def create_mock_model():
    print("Generating a lightweight dummy model for Streamlit UI testing...")
    
    # We need a model that accepts (None, 256, 256, 3) and outputs a single probability
    inputs = tf.keras.Input(shape=(256, 256, 3))
    
    # Very simple lightweight network instead of full EfficientNet (which is 150MB+ and takes time to init/save)
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Compile the model purely so it can be saved properly
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    output_path = "models/saved_model/best_model.h5"
    model.save(output_path)
    
    print(f"Mock model successfully saved to {output_path}")

if __name__ == "__main__":
    create_mock_model()
