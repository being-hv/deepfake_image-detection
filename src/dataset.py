import os
import tensorflow as tf

def get_data_generators(data_dir, batch_size=32, img_size=(256, 256)):
    """
    Creates and returns train, validation, and test tf.data.Datasets.
    Expects data_dir to contain 'train', 'val', and 'test' subdirectories,
    each with 'real' and 'fake' subfolders.
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # Optional logic if paths are not strictly split yet but we want to fall back gracefully
    if not os.path.exists(train_dir):
        print(f"Warning: {train_dir} does not exist. Please organize data as train/val/test.")
        return None, None, None

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='binary',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=42
    )

    val_ds = None
    if os.path.exists(val_dir):
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            labels='inferred',
            label_mode='binary',
            batch_size=batch_size,
            image_size=img_size,
            shuffle=False
        )

    test_ds = None
    if os.path.exists(test_dir):
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels='inferred',
            label_mode='binary',
            batch_size=batch_size,
            image_size=img_size,
            shuffle=False
        )

    return train_ds, val_ds, test_ds

def get_data_augmentation():
    """
    Returns a Keras Sequential model for data augmentation.
    Applies Rotation, Horizontal Flip, Brightness Variation, and Zoom.
    """
    data_augmentation = tf.keras.Sequential([
        # tf.keras.layers.Rescaling is included in the base model instead or applied directly.
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(factor=0.2),
    ], name="data_augmentation")
    return data_augmentation

def prepare_dataset(ds, augment=False):
    """
    Applies augmentation, normalization/rescaling, and prefetching for performance.
    """
    if ds is None:
        return ds

    # Rescaling pixels to [0, 1] if not relying on backbone specific preprocess input
    # Most keras applications backbones expect pixels [0,255] or [-1, 1], 
    # but we will normalize manually or use keras backbones' specific preprocessing later.
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    if augment:
        data_augmentation = get_data_augmentation()
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    
    return ds.prefetch(buffer_size=AUTOTUNE)

if __name__ == "__main__":
    # Test dataset pipeline
    print("Dataset module loaded.")
    aug = get_data_augmentation()
    print("Data augmentation pipeline constructed:", aug.layers)
