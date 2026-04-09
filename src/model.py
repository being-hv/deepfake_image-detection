import tensorflow as tf

def build_model(backbone_name='efficientnetb4', input_shape=(256, 256, 3), fine_tune=False):
    """
    Builds the Deepfake Detection model using transfer learning.
    
    Args:
        backbone_name: 'efficientnetb4', 'resnet50', or 'xception'
        input_shape: Shape of the input images
        fine_tune: If False, the backbone is frozen (warm-up phase).
                   If True, upper layers of the backbone are unfrozen for fine-tuning.
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Select backbone and its specific preprocessing if necessary
    backbone_name = backbone_name.lower()
    
    # We apply specific preprocessing based on the choice
    if backbone_name == 'efficientnetb4':
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        base_model = tf.keras.applications.EfficientNetB4(
            include_top=False, weights='imagenet', input_tensor=x
        )
    elif backbone_name == 'resnet50':
        x = tf.keras.applications.resnet50.preprocess_input(inputs)
        base_model = tf.keras.applications.ResNet50(
            include_top=False, weights='imagenet', input_tensor=x
        )
    elif backbone_name == 'xception':
        x = tf.keras.applications.xception.preprocess_input(inputs)
        base_model = tf.keras.applications.Xception(
            include_top=False, weights='imagenet', input_tensor=x
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
        
    # Freezing logic
    base_model.trainable = fine_tune
    if fine_tune:
        # Freeze the bottom layers and unfreeze the top layers
        # Example for partial unfreezing: unfreeze top 20 layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        for layer in base_model.layers[-20:]:
            # Ensure BatchNormalization layers remain frozen to prevent statistics destruction
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    # Build top layers on top of backbone
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(x)

    model = tf.keras.Model(inputs, outputs, name=f'deepfake_detector_{backbone_name}')
    
    return model

if __name__ == "__main__":
    # Test model construction
    model = build_model(backbone_name='efficientnetb4', fine_tune=False)
    model.summary()
