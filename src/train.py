import os
import argparse
import json
import tensorflow as tf
from dataset import get_data_generators, prepare_dataset
from model import build_model

def train(args):
    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading data from {args.data_dir}...")
    train_ds, val_ds, test_ds = get_data_generators(args.data_dir, batch_size=args.batch_size)
    
    if train_ds is None or val_ds is None:
        raise ValueError("Training and validation datasets must be present.")
        
    train_ds = prepare_dataset(train_ds, augment=True)
    val_ds = prepare_dataset(val_ds, augment=False)
    
    # 2. Build Model for Warm-up (Frozen backbone)
    print(f"Building {args.model} for Warm-up...")
    model = build_model(backbone_name=args.model, fine_tune=False)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='roc_auc')]
    )
    
    # Callbacks
    model_path = os.path.join(args.output_dir, 'best_model.h5')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Run Warm-up Training (Only head is training)
    warmup_epochs = min(5, args.epochs // 3) if args.epochs > 5 else args.epochs
    print(f"Starting Warm-up Phase for {warmup_epochs} epochs...")
    
    history_warmup = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        callbacks=callbacks
    )
    
    # 3. Fine-Tuning Phase
    if args.epochs > warmup_epochs:
        print("Starting Fine-tuning Phase (Unfreezing top layers of backbone)...")
        # Re-build / re-configure model to unfreeze upper layers
        model.trainable = True
        
        # Robustly locate the base_model (backbone) inside the top-level model
        base_model = None
        for layer in model.layers:
            if hasattr(layer, 'layers') and not isinstance(layer, tf.keras.Sequential):
                base_model = layer
                break
                
        if base_model is not None:
            for layer in base_model.layers[:-20]:
                layer.trainable = False
            for layer in base_model.layers[-20:]:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True
        else:
            print("Warning: Could not identify backbone model for partial fine-tuning. Unfreezing entire model.")
                
        # Recompile with a very low learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate / 10),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='roc_auc')]
        )
        
        finetune_epochs = args.epochs - warmup_epochs
        history_finetune = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            initial_epoch=history_warmup.epoch[-1] + 1,
            callbacks=callbacks
        )
        
        # Merge histories
        for k in history_warmup.history.keys():
            history_warmup.history[k].extend(history_finetune.history[k])
            
    # Save training history for Streamlit Dashboard
    history_file = os.path.join(args.output_dir, 'training_history.json')
    # converting float32 to float
    hist_dict = {k: [float(i) for i in v] for k, v in history_warmup.history.items()}
    with open(history_file, 'w') as f:
        json.dump(hist_dict, f)
        
    print(f"Training completed. Model saved at {model_path}")
    print(f"History saved at {history_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deepfake Detection Model")
    parser.add_argument('--data_dir', type=str, default='data/', help='Root directory containing train/, val/, test/')
    parser.add_argument('--epochs', type=int, default=20, help='Total Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--model', type=str, default='efficientnetb4', help='Backbone architecture')
    parser.add_argument('--output_dir', type=str, default='models/saved_model/', help='Where to save checkpoints')
    
    args = parser.parse_args()
    train(args)
